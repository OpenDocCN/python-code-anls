# `.\lucidrains\magvit2-pytorch\magvit2_pytorch\magvit2_pytorch.py`

```py
# 导入必要的库
import copy
from pathlib import Path
from math import log2, ceil, sqrt
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from torch.autograd import grad as torch_grad

import torchvision
from torchvision.models import VGG16_Weights

from collections import namedtuple

# 导入自定义模块
from vector_quantize_pytorch import LFQ, FSQ
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List
from magvit2_pytorch.attend import Attend
from magvit2_pytorch.version import __version__
from gateloop_transformer import SimpleGateLoopLayer
from taylor_series_linear_attention import TaylorSeriesLinearAttn
from kornia.filters import filter3d

import pickle

# helper

# 检查变量是否存在
def exists(v):
    return v is not None

# 返回默认值
def default(v, d):
    return v if exists(v) else d

# 安全获取列表中的元素
def safe_get_index(it, ind, default = None):
    if ind < len(it):
        return it[ind]
    return default

# 将输入转换为元组
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# 返回输入本身
def identity(t, *args, **kwargs):
    return t

# 检查一个数是否可以被另一个数整除
def divisible_by(num, den):
    return (num % den) == 0

# 将输入打包成指定模式
def pack_one(t, pattern):
    return pack([t], pattern)

# 将输入解包成指定模式
def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# 在张量的末尾添加指定维度
def append_dims(t, ndims: int):
    return t.reshape(*t.shape, *((1,) * ndims))

# 检查一个数是否为奇数
def is_odd(n):
    return not divisible_by(n, 2)

# 删除对象的属性
def maybe_del_attr_(o, attr):
    if hasattr(o, attr):
        delattr(o, attr)

# 将输入转换为元组
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# tensor helpers

# 对张量进行 L2 归一化
def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

# 在指定维度上对张量进行填充
def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# 从视频中选择指定帧
def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    batch_indices = torch.arange(batch, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

# gan related

# 计算梯度惩罚
def gradient_penalty(images, output):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = images.device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

# Leaky ReLU 激活函数
def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

# Hinge 损失函数（判别器）
def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

# Hinge 损失函数（生成器）
def hinge_gen_loss(fake):
    return -fake.mean()

# 计算损失对层的梯度
@autocast(enabled = False)
@beartype
def grad_layer_wrt_loss(
    loss: Tensor,
    layer: nn.Parameter
):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# helper decorators

# 移除 VGG 属性
def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

# helper classes

# 顺序模块
def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    return nn.Sequential(*modules)

# 残差模块
class Residual(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn
    # 定义一个前向传播函数，接受输入 x 和其他关键字参数
    def forward(self, x, **kwargs):
        # 调用函数 fn 对输入 x 进行处理，并将结果与输入 x 相加后返回
        return self.fn(x, **kwargs) + x
# 一系列张量操作，将张量转换为 (batch, time, feature dimension) 格式，然后再转回来

class ToTimeSequence(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # 重新排列张量的维度，将其转换为 (batch, ..., feature, channel) 格式
        x = rearrange(x, 'b c f ... -> b ... f c')
        # 打包张量，将其转换为 (batch, ..., feature, channel) 格式
        x, ps = pack_one(x, '* n c')

        # 使用给定的函数对张量进行操作
        o = self.fn(x, **kwargs)

        # 解包张量，将其转换回原始格式
        o = unpack_one(o, ps, '* n c')
        # 重新排列张量的维度，将其转换回原始格式
        return rearrange(o, 'b ... f c -> b c f ...')


class SqueezeExcite(Module):
    # 全局上下文网络 - 基于注意力机制的 Squeeze-Excite 变种 (https://arxiv.org/abs/2012.13375)

    def __init__(
        self,
        dim,
        *,
        dim_out = None,
        dim_hidden_min = 16,
        init_bias = -10
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        # 创建卷积层，用于计算注意力权重
        self.to_k = nn.Conv2d(dim, 1, 1)
        dim_hidden = max(dim_hidden_min, dim_out // 2)

        # 创建包含卷积层和激活函数的网络结构
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim_hidden, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(dim_hidden, dim_out, 1),
            nn.Sigmoid()
        )

        # 初始化网络参数
        nn.init.zeros_(self.net[-2].weight)
        nn.init.constant_(self.net[-2].bias, init_bias)

    def forward(self, x):
        orig_input, batch = x, x.shape[0]
        is_video = x.ndim == 5

        if is_video:
            # 重新排列视频张量的维度
            x = rearrange(x, 'b c f h w -> (b f) c h w')

        # 计算上下文信息
        context = self.to_k(x)

        # 计算注意力权重
        context = rearrange(context, 'b c h w -> b c (h w)').softmax(dim = -1)
        spatial_flattened_input = rearrange(x, 'b c h w -> b c (h w)')

        # 使用注意力权重对输入进行加权求和
        out = einsum('b i n, b c n -> b c i', context, spatial_flattened_input)
        out = rearrange(out, '... -> ... 1')
        gates = self.net(out)

        if is_video:
            # 将结果转换回视频张量的格式
            gates = rearrange(gates, '(b f) c h w -> b c f h w', b = batch)

        return gates * orig_input

# token shifting

class TokenShift(Module):
    @beartype
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        # 将输入张量分成两部分
        x, x_shift = x.chunk(2, dim = 1)
        # 在时间维度上进行填充，实现时间维度的位移
        x_shift = pad_at_dim(x_shift, (1, -1), dim = 2)
        # 将两部分张量连接起来
        x = torch.cat((x, x_shift), dim = 1)
        return self.fn(x, **kwargs)

# rmsnorm

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        # 对输入张量进行 RMS 归一化
        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias

class AdaptiveRMSNorm(Module):
    def __init__(
        self,
        dim,
        *,
        dim_cond,
        channel_first = False,
        images = False,
        bias = False
    ):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.dim_cond = dim_cond
        self.channel_first = channel_first
        self.scale = dim ** 0.5

        # 创建线性层，用于生成 gamma 和 bias
        self.to_gamma = nn.Linear(dim_cond, dim)
        self.to_bias = nn.Linear(dim_cond, dim) if bias else None

        # 初始化线性层参数
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)

        if bias:
            nn.init.zeros_(self.to_bias.weight)
            nn.init.zeros_(self.to_bias.bias)

    @beartype
    # 定义一个前向传播函数，接受输入张量 x 和条件张量 cond
    def forward(self, x: Tensor, *, cond: Tensor):
        # 获取批量大小
        batch = x.shape[0]
        # 断言条件张量的形状为 (batch, self.dim_cond)
        assert cond.shape == (batch, self.dim_cond)

        # 根据条件张量生成 gamma
        gamma = self.to_gamma(cond)

        # 初始化偏置为 0
        bias = 0.
        # 如果存在偏置生成函数
        if exists(self.to_bias):
            # 根据条件张量生成偏置
            bias = self.to_bias(cond)

        # 如果通道在前
        if self.channel_first:
            # 在 gamma 的维度前面添加维度，使其与输入张量 x 的维度相同
            gamma = append_dims(gamma, x.ndim - 2)

            # 如果存在偏置生成函数
            if exists(self.to_bias):
                # 在偏置的维度前面添加维度，使其与输入张量 x 的维度相同
                bias = append_dims(bias, x.ndim - 2)

        # 对输入张量 x 进行归一化，根据通道顺序选择归一化的维度，然后乘以缩放因子 scale 和 gamma，最后加上偏置 bias
        return F.normalize(x, dim = (1 if self.channel_first else -1)) * self.scale * gamma + bias
# 定义一个名为 Attention 的类，继承自 Module 类
class Attention(Module):
    # 初始化函数，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: Optional[int] = None,
        causal = False,
        dim_head = 32,
        heads = 8,
        flash = False,
        dropout = 0.,
        num_memory_kv = 4
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算内部维度
        dim_inner = dim_head * heads

        # 检查是否需要条件
        self.need_cond = exists(dim_cond)

        # 根据是否需要条件选择不同的归一化方法
        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond = dim_cond)
        else:
            self.norm = RMSNorm(dim)

        # 构建 QKV 网络
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        # 断言内存键值对数量大于 0
        assert num_memory_kv > 0
        # 初始化内存键值对
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_memory_kv, dim_head))

        # 构建 Attend 层
        self.attend = Attend(
            causal = causal,
            dropout = dropout,
            flash = flash
        )

        # 构建输出层
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias = False)
        )

    # 前向传播函数
    @beartype
    def forward(
        self,
        x,
        mask: Optional[Tensor ] = None,
        cond: Optional[Tensor] = None
    ):
        # 根据是否需要条件选择不同的参数
        maybe_cond_kwargs = dict(cond = cond) if self.need_cond else dict()

        # 对输入进行归一化
        x = self.norm(x, **maybe_cond_kwargs)

        # 获取 QKV
        q, k, v = self.to_qkv(x)

        # 重复内存键值对
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = q.shape[0]), self.mem_kv)
        k = torch.cat((mk, k), dim = -2)
        v = torch.cat((mv, v), dim = -2)

        # 进行注意力计算
        out = self.attend(q, k, v, mask = mask)
        return self.to_out(out)

# 定义一个名为 LinearAttention 的类，继承自 Module 类
class LinearAttention(Module):
    """
    using the specific linear attention proposed in https://arxiv.org/abs/2106.09681
    """

    # 初始化函数，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        dim,
        dim_cond: Optional[int] = None,
        dim_head = 8,
        heads = 8,
        dropout = 0.
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 计算内部维度
        dim_inner = dim_head * heads

        # 检查是否需要条件
        self.need_cond = exists(dim_cond)

        # 根据是否需要条件选择不同的归一化方法
        if self.need_cond:
            self.norm = AdaptiveRMSNorm(dim, dim_cond = dim_cond)
        else:
            self.norm = RMSNorm(dim)

        # 构建 TaylorSeriesLinearAttn 层
        self.attn = TaylorSeriesLinearAttn(
            dim = dim,
            dim_head = dim_head,
            heads = heads
        )

    # 前向传播函数
    def forward(
        self,
        x,
        cond: Optional[Tensor] = None
    ):
        # 根据是否需要条件选择不同的参数
        maybe_cond_kwargs = dict(cond = cond) if self.need_cond else dict()

        # 对输入进行归一化
        x = self.norm(x, **maybe_cond_kwargs)

        return self.attn(x)

# 定义一个名为 LinearSpaceAttention 的类，继承自 LinearAttention 类
class LinearSpaceAttention(LinearAttention):
    # 重写前向传播函数
    def forward(self, x, *args, **kwargs):
        # 重新排列输入数据
        x = rearrange(x, 'b c ... h w -> b ... h w c')
        x, batch_ps = pack_one(x, '* h w c')
        x, seq_ps = pack_one(x, 'b * c')

        # 调用父类的前向传播函数
        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, 'b * c')
        x = unpack_one(x, batch_ps, '* h w c')
        return rearrange(x, 'b ... h w c -> b c ... h w')

# 定义一个名为 SpaceAttention 的类，继承自 Attention 类
class SpaceAttention(Attention):
    # 重写前向传播函数
    def forward(self, x, *args, **kwargs):
        # 重新排列输入数据
        x = rearrange(x, 'b c t h w -> b t h w c')
        x, batch_ps = pack_one(x, '* h w c')
        x, seq_ps = pack_one(x, 'b * c')

        # 调用父类的前向传播函数
        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, seq_ps, 'b * c')
        x = unpack_one(x, batch_ps, '* h w c')
        return rearrange(x, 'b t h w c -> b c t h w')

# 定义一个名为 TimeAttention 的类，继承自 Attention 类
class TimeAttention(Attention):
    # 重写前向传播函数
    def forward(self, x, *args, **kwargs):
        # 重新排列输入数据
        x = rearrange(x, 'b c t h w -> b h w t c')
        x, batch_ps = pack_one(x, '* t c')

        # 调用父类的前向传播函数
        x = super().forward(x, *args, **kwargs)

        x = unpack_one(x, batch_ps, '* t c')
        return rearrange(x, 'b h w t c -> b c t h w')

# 定义一个名为 GEGLU 的类，继承自 Module 类
class GEGLU(Module):
    # 前向传播函数
    def forward(self, x):
        # 将输入数据分成两部分
        x, gate = x.chunk(2, dim = 1)
        return F.gelu(gate) * x

# 定义一个名为 FeedForward 的类，继承自 Module 类
class FeedForward(Module):
    @beartype
    # 初始化函数，设置神经网络的参数
    def __init__(
        self,
        dim,  # 输入数据的维度
        *,
        dim_cond: Optional[int] = None,  # 条件维度，默认为None
        mult = 4,  # 倍数，默认为4
        images = False  # 是否为图像数据，默认为False
    ):
        super().__init__()  # 调用父类的初始化函数
        # 根据是否为图像数据选择不同的卷积层类
        conv_klass = nn.Conv2d if images else nn.Conv3d

        # 根据条件维度是否存在选择不同的归一化层类
        rmsnorm_klass = RMSNorm if not exists(dim_cond) else partial(AdaptiveRMSNorm, dim_cond = dim_cond)

        # 创建可能的自适应归一化层类
        maybe_adaptive_norm_klass = partial(rmsnorm_klass, channel_first = True, images = images)

        # 计算内部维度
        dim_inner = int(dim * mult * 2 / 3)

        # 初始化归一化层
        self.norm = maybe_adaptive_norm_klass(dim)

        # 初始化神经网络结构
        self.net = Sequential(
            conv_klass(dim, dim_inner * 2, 1),  # 卷积层
            GEGLU(),  # 激活函数
            conv_klass(dim_inner, dim, 1)  # 卷积层
        )

    # 前向传播函数
    @beartype
    def forward(
        self,
        x: Tensor,  # 输入数据张量
        *,
        cond: Optional[Tensor] = None  # 条件张量，默认为None
    ):
        # 根据条件张量是否存在选择不同的参数
        maybe_cond_kwargs = dict(cond = cond) if exists(cond) else dict()

        # 对输入数据进行归一化处理
        x = self.norm(x, **maybe_cond_kwargs)
        return self.net(x)  # 返回神经网络处理后的结果
# 定义一个带有反锯齿下采样的鉴别器（模糊池 Zhang 等人）

class Blur(Module):
    def __init__(self):
        super().__init__()
        # 定义一个张量 f
        f = torch.Tensor([1, 2, 1])
        # 将张量 f 注册为缓冲区
        self.register_buffer('f', f)

    def forward(
        self,
        x,
        space_only = False,
        time_only = False
    ):
        # 断言空间和时间只能选择一个
        assert not (space_only and time_only)

        # 获取张量 f
        f = self.f

        if space_only:
            # 对 f 进行乘法操作
            f = einsum('i, j -> i j', f, f)
            # 重新排列张量 f
            f = rearrange(f, '... -> 1 1 ...')
        elif time_only:
            # 重新排列张量 f
            f = rearrange(f, 'f -> 1 f 1 1')
        else:
            # 对 f 进行乘法操作
            f = einsum('i, j, k -> i j k', f, f, f)
            # 重新排列张量 f
            f = rearrange(f, '... -> 1 ...')

        # 判断输入 x 是否为图像
        is_images = x.ndim == 4

        if is_images:
            # 重新排列输入 x
            x = rearrange(x, 'b c h w -> b c 1 h w')

        # 对输入 x 进行 3D 滤波
        out = filter3d(x, f, normalized = True)

        if is_images:
            # 重新排列输出 out
            out = rearrange(out, 'b c 1 h w -> b c h w')

        return out

class DiscriminatorBlock(Module):
    def __init__(
        self,
        input_channels,
        filters,
        downsample = True,
        antialiased_downsample = True
    ):
        super().__init__()
        # 定义卷积层 conv_res
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        # 定义神经网络结构 net
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding = 1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding = 1),
            leaky_relu()
        )

        # 如果需要反锯齿下采样，则定义模糊层 maybe_blur
        self.maybe_blur = Blur() if antialiased_downsample else None

        # 如果需要下采样，则定义下采样层 downsample
        self.downsample = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
            nn.Conv2d(filters * 4, filters, 1)
        ) if downsample else None

    def forward(self, x):
        # 对输入 x 进行卷积操作，得到 res
        res = self.conv_res(x)

        # 对输入 x 进行神经网络结构操作
        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                # 如果存在模糊层，则对 x 进行模糊操作
                x = self.maybe_blur(x, space_only = True)

            # 对 x 进行下采样操作
            x = self.downsample(x)

        # 对 x 进行加权求和并缩放操作
        x = (x + res) * (2 ** -0.5)
        return x

class Discriminator(Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        max_dim = 512,
        attn_heads = 8,
        attn_dim_head = 32,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        ff_mult = 4,
        antialiased_downsample = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 将图像大小转换为元组
        image_size = pair(image_size)
        # 计算图像分辨率的最小值
        min_image_resolution = min(image_size)

        # 计算层数
        num_layers = int(log2(min_image_resolution) - 2)

        blocks = []

        # 计算每一层的维度
        layer_dims = [channels] + [(dim * 4) * (2 ** i) for i in range(num_layers + 1)]
        # 将每一层的维度限制在最大维度内
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        # 将每一层的输入输出维度组成元组
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            # 创建判别器块
            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample = is_not_last,
                antialiased_downsample = antialiased_downsample
            )

            # 创建注意力块
            attn_block = Sequential(
                Residual(LinearSpaceAttention(
                    dim = out_chan,
                    heads = linear_attn_heads,
                    dim_head = linear_attn_dim_head
                )),
                Residual(FeedForward(
                    dim = out_chan,
                    mult = ff_mult,
                    images = True
                ))
            )

            blocks.append(ModuleList([
                block,
                attn_block
            ]))

            image_resolution //= 2

        self.blocks = ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2 ** num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        # 定义输出层
        self.to_logits = Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding = 1),
            leaky_relu(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )

    def forward(self, x):

        # 遍历每个块和注意力块
        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)

        return self.to_logits(x)
# 定义一个继承自 Module 的类 Conv3DMod，用于实现可调制的卷积，用于在潜变量上进行条件化
class Conv3DMod(Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        dim,
        *,
        spatial_kernel,
        time_kernel,
        causal = True,
        dim_out = None,
        demod = True,
        eps = 1e-8,
        pad_mode = 'zeros'
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.eps = eps

        # 断言空间和时间卷积核为奇数
        assert is_odd(spatial_kernel) and is_odd(time_kernel)

        self.spatial_kernel = spatial_kernel
        self.time_kernel = time_kernel

        # 根据是否因果，设置时间填充
        time_padding = (time_kernel - 1, 0) if causal else ((time_kernel // 2,) * 2)

        self.pad_mode = pad_mode
        self.padding = (*((spatial_kernel // 2,) * 4), *time_padding)
        self.weights = nn.Parameter(torch.randn((dim_out, dim, time_kernel, spatial_kernel, spatial_kernel)))

        self.demod = demod

        # 初始化权重
        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'selu')

    # 前向传播函数
    @beartype
    def forward(
        self,
        fmap,
        cond: Tensor
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        b = fmap.shape[0]

        # 准备用于调制的权重
        weights = self.weights

        # 进行调制和解调制，类似 stylegan2 中的操作
        cond = rearrange(cond, 'b i -> b 1 i 1 1 1')

        weights = weights * (cond + 1)

        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k0 k1 k2 -> b o 1 1 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c t h w -> 1 (b c) t h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        fmap = F.pad(fmap, self.padding, mode = self.pad_mode)
        fmap = F.conv3d(fmap, weights, groups = b)

        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)

# 定义一个继承自 Module 的类 SpatialDownsample2x，用于进行空间下采样
class SpatialDownsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, stride = 2, padding = kernel_size // 2)

    # 前向传播函数
    def forward(self, x):
        x = self.maybe_blur(x, space_only = True)

        x = rearrange(x, 'b c t h w -> b t c h w')
        x, ps = pack_one(x, '* c h w')

        out = self.conv(x)

        out = unpack_one(out, ps, '* c h w')
        out = rearrange(out, 'b t c h w -> b c t h w')
        return out

# 定义一个继承自 Module 的类 TimeDownsample2x，用于进行时间下采样
class TimeDownsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        self.maybe_blur = Blur() if antialias else identity
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride = 2)

    # 前向传播函数
    def forward(self, x):
        x = self.maybe_blur(x, time_only = True)

        x = rearrange(x, 'b c t h w -> b h w c t')
        x, ps = pack_one(x, '* c t')

        x = F.pad(x, self.time_causal_padding)
        out = self.conv(x)

        out = unpack_one(out, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out

# 定义一个继承自 Module 的类 SpatialUpsample2x，用于进行空间上采样
class SpatialUpsample2x(Module):
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange('b (c p1 p2) h w -> b c (h p1) (w p2)', p1 = 2, p2 = 2)
        )

        self.init_conv_(conv)
    # 初始化卷积层的权重和偏置
    def init_conv_(self, conv):
        # 获取卷积层的输出通道数、输入通道数、高度和宽度
        o, i, h, w = conv.weight.shape
        # 创建一个与卷积层权重相同形状的张量
        conv_weight = torch.empty(o // 4, i, h, w)
        # 使用 Kaiming 初始化方法初始化权重
        nn.init.kaiming_uniform_(conv_weight)
        # 将权重张量重复4次，扩展为4倍的输出通道数
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        # 将初始化好的权重复制给卷积层的权重
        conv.weight.data.copy_(conv_weight)
        # 初始化卷积层的偏置为零
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        # 重新排列输入张量的维度，将通道维度放到第二个位置
        x = rearrange(x, 'b c t h w -> b t c h w')
        # 将输入张量打包成一个元组，每个元素为一个通道的数据
        x, ps = pack_one(x, '* c h w')

        # 将打包后的输入张量传入网络进行前向传播
        out = self.net(x)

        # 将网络输出解包，恢复为原始形状
        out = unpack_one(out, ps, '* c h w')
        # 重新排列输出张量的维度，将通道维度放回最后一个位置
        out = rearrange(out, 'b t c h w -> b c t h w')
        # 返回前向传播结果
        return out
# 定义一个类 TimeUpsample2x，继承自 Module 类
class TimeUpsample2x(Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        dim_out = None
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 如果未指定输出维度，则默认与输入维度相同
        dim_out = default(dim_out, dim)
        # 创建一个 1 维卷积层，输入维度为 dim，输出维度为 dim_out * 2，卷积核大小为 1
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        # 使用 nn.Sequential 定义网络结构
        self.net = nn.Sequential(
            conv,
            nn.SiLU(),  # 使用 SiLU 激活函数
            Rearrange('b (c p) t -> b c (t p)', p = 2)  # 重新排列张量维度
        )

        # 初始化卷积层的权重
        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        # 创建一个与卷积层权重相同形状的张量
        conv_weight = torch.empty(o // 2, i, t)
        # 使用 kaiming_uniform_ 方法初始化权重
        nn.init.kaiming_uniform_(conv_weight)
        # 将权重张量重复一次
        conv_weight = repeat(conv_weight, 'o ... -> (o 2) ...')

        # 将初始化后的权重赋值给卷积层
        conv.weight.data.copy_(conv_weight)
        # 将偏置项初始化为零
        nn.init.zeros_(conv.bias.data)

    # 前向传播函数
    def forward(self, x):
        # 重新排列输入张量的维度
        x = rearrange(x, 'b c t h w -> b h w c t')
        # 打包输入张量
        x, ps = pack_one(x, '* c t')

        # 网络前向传播
        out = self.net(x)

        # 解包输出张量
        out = unpack_one(out, ps, '* c t')
        # 重新排列输出张量的维度
        out = rearrange(out, 'b h w c t -> b c t h w')
        return out

# 定义一个函数 SameConv2d，用于创建相同维度的二维卷积层
def SameConv2d(dim_in, dim_out, kernel_size):
    kernel_size = cast_tuple(kernel_size, 2)
    padding = [k // 2 for k in kernel_size]
    return nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding)

# 定义一个类 CausalConv3d，继承自 Module 类
class CausalConv3d(Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode = 'constant',
        **kwargs
    ):
        # 调用父类的初始化函数
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 确保高度和宽度的卷积核大小为奇数
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop('dilation', 1)
        stride = kwargs.pop('stride', 1)

        # 设置时间维度的填充大小
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.pad_mode = pad_mode
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        # 创建一个三维卷积层
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    # 前向传播函数
    def forward(self, x):
        # 根据填充模式选择填充方式
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else 'constant'

        # 对输入张量进行填充
        x = F.pad(x, self.time_causal_padding, mode = pad_mode)
        return self.conv(x)

# 定义一个函数 ResidualUnit，用于创建残差单元
@beartype
def ResidualUnit(
    dim,
    kernel_size: Union[int, Tuple[int, int, int]],
    pad_mode: str = 'constant'
):
    # 构建残差单元网络结构
    net = Sequential(
        CausalConv3d(dim, dim, kernel_size, pad_mode = pad_mode),
        nn.ELU(),  # 使用 ELU 激活函数
        nn.Conv3d(dim, dim, 1),
        nn.ELU(),
        SqueezeExcite(dim)
    )

    return Residual(net)

# 定义一个类 ResidualUnitMod，继承自 Module 类
@beartype
class ResidualUnitMod(Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        kernel_size: Union[int, Tuple[int, int, int]],
        *,
        dim_cond,
        pad_mode: str = 'constant',
        demod = True
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert height_kernel_size == width_kernel_size

        # 线性层，用于将条件信息转换为相同维度
        self.to_cond = nn.Linear(dim_cond, dim)

        # 创建一个 Conv3DMod 层
        self.conv = Conv3DMod(
            dim = dim,
            spatial_kernel = height_kernel_size,
            time_kernel = time_kernel_size,
            causal = True,
            demod = demod,
            pad_mode = pad_mode
        )

        # 创建一个 1x1x1 三维卷积层
        self.conv_out = nn.Conv3d(dim, dim, 1)

    # 前向传播函数
    @beartype
    def forward(
        self,
        x,
        cond: Tensor,
    ):
        res = x
        cond = self.to_cond(cond)

        # 进行卷积操作
        x = self.conv(x, cond = cond)
        x = F.elu(x)
        x = self.conv_out(x)
        x = F.elu(x)
        return x + res

# 定义一个类 CausalConvTranspose3d，继承自 Module 类
    # 初始化函数，定义了一个卷积转置层
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        *,
        time_stride,
        **kwargs
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 将 kernel_size 转换为三元组
        kernel_size = cast_tuple(kernel_size, 3)

        # 分别获取时间、高度和宽度的卷积核大小
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        # 断言高度卷积核大小和宽度卷积核大小为奇数
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 设置上采样因子为时间步长
        self.upsample_factor = time_stride

        # 计算高度和宽度的填充值
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        # 设置步长和填充值
        stride = (time_stride, 1, 1)
        padding = (0, height_pad, width_pad)

        # 创建一个三维卷积转置层
        self.conv = nn.ConvTranspose3d(chan_in, chan_out, kernel_size, stride, padding = padding, **kwargs)

    # 前向传播函数
    def forward(self, x):
        # 断言输入张量 x 的维度为 5
        assert x.ndim == 5
        # 获取时间维度的大小
        t = x.shape[2]

        # 对输入张量进行卷积转置操作
        out = self.conv(x)

        # 裁剪输出张量的时间维度，保留 t * 上采样因子 个时间步
        out = out[..., :(t * self.upsample_factor), :, :]
        # 返回处理后的输出张量
        return out
# 定义了 LossBreakdown 命名元组，包含了不同损失的分解信息
LossBreakdown = namedtuple('LossBreakdown', [
    'recon_loss',
    'lfq_aux_loss',
    'quantizer_loss_breakdown',
    'perceptual_loss',
    'adversarial_gen_loss',
    'adaptive_adversarial_weight',
    'multiscale_gen_losses',
    'multiscale_gen_adaptive_weights'
])

# 定义了 DiscrLossBreakdown 命名元组，包含了鉴别器损失的分解信息
DiscrLossBreakdown = namedtuple('DiscrLossBreakdown', [
    'discr_loss',
    'multiscale_discr_losses',
    'gradient_penalty'
])

# 定义了 VideoTokenizer 类，继承自 Module 类
class VideoTokenizer(Module):
    # 初始化方法
    @beartype
    def __init__(
        self,
        *,
        image_size,
        layers: Tuple[Union[str, Tuple[str, int]], ...] = (
            'residual',
            'residual',
            'residual'
        ),
        residual_conv_kernel_size = 3,
        num_codebooks = 1,
        codebook_size: Optional[int] = None,
        channels = 3,
        init_dim = 64,
        max_dim = float('inf'),
        dim_cond = None,
        dim_cond_expansion_factor = 4.,
        input_conv_kernel_size: Tuple[int, int, int] = (7, 7, 7),
        output_conv_kernel_size: Tuple[int, int, int] = (3, 3, 3),
        pad_mode: str = 'constant',
        lfq_entropy_loss_weight = 0.1,
        lfq_commitment_loss_weight = 1.,
        lfq_diversity_gamma = 2.5,
        quantizer_aux_loss_weight = 1.,
        lfq_activation = nn.Identity(),
        use_fsq = False,
        fsq_levels: Optional[List[int]] = None,
        attn_dim_head = 32,
        attn_heads = 8,
        attn_dropout = 0.,
        linear_attn_dim_head = 8,
        linear_attn_heads = 16,
        vgg: Optional[Module] = None,
        vgg_weights: VGG16_Weights = VGG16_Weights.DEFAULT,
        perceptual_loss_weight = 1e-1,
        discr_kwargs: Optional[dict] = None,
        multiscale_discrs: Tuple[Module, ...] = tuple(),
        use_gan = True,
        adversarial_loss_weight = 1.,
        grad_penalty_loss_weight = 10.,
        multiscale_adversarial_loss_weight = 1.,
        flash_attn = True,
        separate_first_frame_encoding = False
    # 返回属性 device，返回 zero 属性的设备信息
    @property
    def device(self):
        return self.zero.device

    # 类方法，初始化并从路径加载模型
    @classmethod
    def init_and_load_from(cls, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')

        assert 'config' in pkg, 'model configs were not found in this saved checkpoint'

        config = pickle.loads(pkg['config'])
        tokenizer = cls(**config)
        tokenizer.load(path, strict = strict)
        return tokenizer

    # 返回模型参数
    def parameters(self):
        return [
            *self.conv_in.parameters(),
            *self.conv_in_first_frame.parameters(),
            *self.conv_out_first_frame.parameters(),
            *self.conv_out.parameters(),
            *self.encoder_layers.parameters(),
            *self.decoder_layers.parameters(),
            *self.encoder_cond_in.parameters(),
            *self.decoder_cond_in.parameters(),
            *self.quantizers.parameters()
        ]

    # 返回鉴别器参数
    def discr_parameters(self):
        return self.discr.parameters()

    # 复制模型用于评估
    def copy_for_eval(self):
        device = self.device
        vae_copy = copy.deepcopy(self.cpu())

        maybe_del_attr_(vae_copy, 'discr')
        maybe_del_attr_(vae_copy, 'vgg')
        maybe_del_attr_(vae_copy, 'multiscale_discrs')

        vae_copy.eval()
        return vae_copy.to(device)

    # 返回模型状态字典
    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    # 加载模型状态字典
    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    # 保存模型
    def save(self, path, overwrite = True):
        path = Path(path)
        assert overwrite or not path.exists(), f'{str(path)} already exists'

        pkg = dict(
            model_state_dict = self.state_dict(),
            version = __version__,
            config = self._configs
        )

        torch.save(pkg, str(path))
    # 加载模型参数
    def load(self, path, strict = True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()

        # 加载模型参数
        pkg = torch.load(str(path))
        state_dict = pkg.get('model_state_dict')
        version = pkg.get('version')

        # 断言模型参数存在
        assert exists(state_dict)

        # 如果版本信息存在，则打印加载的 tokenizer 版本信息
        if exists(version):
            print(f'loading checkpointed tokenizer from version {version}')

        # 加载模型参数到当前模型
        self.load_state_dict(state_dict, strict = strict)

    # 编码视频
    @beartype
    def encode(
        self,
        video: Tensor,
        quantize = False,
        cond: Optional[Tensor] = None,
        video_contains_first_frame = True
    ):
        # 是否单独编码第一帧
        encode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        # 是否填充视频
        if video_contains_first_frame:
            video_len = video.shape[2]

            video = pad_at_dim(video, (self.time_padding, 0), value = 0., dim = 2)
            video_packed_shape = [torch.Size([self.time_padding]), torch.Size([]), torch.Size([video_len - 1])]

        # 条件编码
        assert (not self.has_cond) or exists(cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (video.shape[0], self.dim_cond)

            cond = self.encoder_cond_in(cond)
            cond_kwargs = dict(cond = cond)

        # 初始卷积
        if encode_first_frame_separately:
            pad, first_frame, video = unpack(video, video_packed_shape, 'b c * h w')
            first_frame = self.conv_in_first_frame(first_frame)

        video = self.conv_in(video)

        if encode_first_frame_separately:
            video, _ = pack([first_frame, video], 'b c * h w')
            video = pad_at_dim(video, (self.time_padding, 0), dim = 2)

        # 编码器层
        for fn, has_cond in zip(self.encoder_layers, self.has_cond_across_layers):

            layer_kwargs = dict()

            if has_cond:
                layer_kwargs = cond_kwargs

            video = fn(video, **layer_kwargs)

        maybe_quantize = identity if not quantize else self.quantizers

        return maybe_quantize(video)

    # 从编码索引解码
    @beartype
    def decode_from_code_indices(
        self,
        codes: Tensor,
        cond: Optional[Tensor] = None,
        video_contains_first_frame = True
    ):
        assert codes.dtype in (torch.long, torch.int32)

        if codes.ndim == 2:
            video_code_len = codes.shape[-1]
            assert divisible_by(video_code_len, self.fmap_size ** 2), f'flattened video ids must have a length ({video_code_len}) that is divisible by the fmap size ({self.fmap_size}) squared ({self.fmap_size ** 2})'

            codes = rearrange(codes, 'b (f h w) -> b f h w', h = self.fmap_size, w = self.fmap_size)

        quantized = self.quantizers.indices_to_codes(codes)

        return self.decode(quantized, cond = cond, video_contains_first_frame = video_contains_first_frame)

    # 解码
    @beartype
    def decode(
        self,
        quantized: Tensor,
        cond: Optional[Tensor] = None,
        video_contains_first_frame = True
        ):
        # 检查是否需要单独解码第一帧
        decode_first_frame_separately = self.separate_first_frame_encoding and video_contains_first_frame

        # 获取批量大小
        batch = quantized.shape[0]

        # 条件输入，如果需要的话
        assert (not self.has_cond) or exists(cond), '`cond` must be passed into tokenizer forward method since conditionable layers were specified'

        if exists(cond):
            assert cond.shape == (batch, self.dim_cond)

            # 将条件输入传入条件编码器
            cond = self.decoder_cond_in(cond)
            cond_kwargs = dict(cond = cond)

        # 解码器层

        x = quantized

        for fn, has_cond in zip(self.decoder_layers, reversed(self.has_cond_across_layers)):

            layer_kwargs = dict()

            if has_cond:
                layer_kwargs = cond_kwargs

            # 逐层解码
            x = fn(x, **layer_kwargs)

        # 转换为像素

        if decode_first_frame_separately:
            left_pad, xff, x = x[:, :, :self.time_padding], x[:, :, self.time_padding], x[:, :, (self.time_padding + 1):]

            # 对输出进行卷积
            out = self.conv_out(x)
            outff = self.conv_out_first_frame(xff)

            # 将第一帧和其余帧打包
            video, _ = pack([outff, out], 'b c * h w')

        else:
            # 对输出进行卷积
            video = self.conv_out(x)

            # 如果视频包含第一帧，则移除填充
            if video_contains_first_frame:
                video = video[:, :, self.time_padding:]

        return video

    @torch.no_grad()
    def tokenize(self, video):
        # 设置为评估模式
        self.eval()
        return self.forward(video, return_codes = True)

    @beartype
    def forward(
        self,
        video_or_images: Tensor,
        cond: Optional[Tensor] = None,
        return_loss = False,
        return_codes = False,
        return_recon = False,
        return_discr_loss = False,
        return_recon_loss_only = False,
        apply_gradient_penalty = True,
        video_contains_first_frame = True,
        adversarial_loss_weight = None,
        multiscale_adversarial_loss_weight = None
# 主要类定义

class MagViT2(Module):
    # 初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()

    # 前向传播方法
    def forward(self, x):
        # 返回输入数据 x，即不做任何处理
        return x
```