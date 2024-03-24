# `.\lucidrains\gigagan-pytorch\gigagan_pytorch\gigagan_pytorch.py`

```
# 导入必要的库
from collections import namedtuple
from pathlib import Path
from math import log2, sqrt
from random import random
from functools import partial

from torchvision import utils

import torch
import torch.nn.functional as F
from torch import nn, einsum, Tensor
from torch.autograd import grad as torch_grad
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from beartype import beartype
from beartype.typing import List, Optional, Tuple, Dict, Union, Iterable

from einops import rearrange, pack, unpack, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

from kornia.filters import filter2d

from ema_pytorch import EMA

from gigagan_pytorch.version import __version__
from gigagan_pytorch.open_clip import OpenClipAdapter
from gigagan_pytorch.optimizer import get_optimizer
from gigagan_pytorch.distributed import all_gather

from tqdm import tqdm

from numerize import numerize

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 检查数组是否为空
@beartype
def is_empty(arr: Iterable):
    return len(arr) == 0

# 返回第一个非空值
def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

# 将输入转换为元组
def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

# 检查数字是否为2的幂
def is_power_of_two(n):
    return log2(n).is_integer()

# 安全地从数组中取出第一个元素
def safe_unshift(arr):
    if len(arr) == 0:
        return None
    return arr.pop(0)

# 检查数字是否可以被另一个数字整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 将数组按照指定数量分组
def group_by_num_consecutive(arr, num):
    out = []
    for ind, el in enumerate(arr):
        if ind > 0 and divisible_by(ind, num):
            yield out
            out = []

        out.append(el)

    if len(out) > 0:
        yield out

# 检查数组中的元素是否唯一
def is_unique(arr):
    return len(set(arr)) == len(arr)

# 无限循环生成数据
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 将数字分成指定数量的组
def num_to_groups(num, divisor):
    groups, remainder = divmod(num, divisor)
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 如果路径不存在，则创建目录
def mkdir_if_not_exists(path):
    path.mkdir(exist_ok = True, parents = True)

# 设置模型参数是否需要梯度
@beartype
def set_requires_grad_(
    m: nn.Module,
    requires_grad: bool
):
    for p in m.parameters():
        p.requires_grad = requires_grad

# 激活函数

# Leaky ReLU 激活函数
def leaky_relu(neg_slope = 0.2):
    return nn.LeakyReLU(neg_slope)

# 创建 3x3 的卷积层
def conv2d_3x3(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 3, padding = 1)

# 张量操作辅助函数

# 对张量取对数
def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

# 计算梯度惩罚
def gradient_penalty(
    images,
    outputs,
    grad_output_weights = None,
    weight = 10,
    scaler: Optional[GradScaler] = None,
    eps = 1e-4
):
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    if exists(scaler):
        outputs = [*map(scaler.scale, outputs)]

    if not exists(grad_output_weights):
        grad_output_weights = (1,) * len(outputs)

    maybe_scaled_gradients, *_ = torch_grad(
        outputs = outputs,
        inputs = images,
        grad_outputs = [(torch.ones_like(output) * weight) for output, weight in zip(outputs, grad_output_weights)],
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )

    gradients = maybe_scaled_gradients

    if exists(scaler):
        scale = scaler.get_scale()
        inv_scale = 1. / max(scale, eps)
        gradients = maybe_scaled_gradients * inv_scale

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

# Hinge GAN 损失函数

# 生成器的 Hinge 损失
def generator_hinge_loss(fake):
    return fake.mean()

# 判别器的 Hinge 损失
def discriminator_hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

# 辅助损失函数

# 辅助匹配损失
def aux_matching_loss(real, fake):
    """
    # 计算负对数似然损失，因为在这个框架中，鉴别器对于真实数据为0，对于生成数据为高值。GANs可以任意交换这一点，只要生成器和鉴别器是对立的即可
    """
    # 返回真实数据和生成数据的负对数似然损失的均值
    return (log(1 + (-real).exp()) + log(1 + (-fake).exp())).mean()
# 使用装饰器 @beartype 对 aux_clip_loss 函数进行类型检查
@beartype
# 定义函数 aux_clip_loss，接受 OpenClipAdapter 类型的 clip 对象、Tensor 类型的 images 和可选的 List[str] 类型的 texts 或 Tensor 类型的 text_embeds
def aux_clip_loss(
    clip: OpenClipAdapter,
    images: Tensor,
    texts: Optional[List[str]] = None,
    text_embeds: Optional[Tensor] = None
):
    # 断言 texts 和 text_embeds 中只有一个存在
    assert exists(texts) ^ exists(text_embeds)

    # 将 images 在所有进程中进行收集
    images, batch_sizes = all_gather(images, 0, None)

    # 如果存在 texts，则使用 clip 对象的 embed_texts 方法获取 text_embeds，并在所有进程中进行收集
    if exists(texts):
        text_embeds, _ = clip.embed_texts(texts)
        text_embeds, _ = all_gather(text_embeds, 0, batch_sizes)

    # 返回 clip 对象的 contrastive_loss 方法计算的损失值
    return clip.contrastive_loss(images = images, text_embeds = text_embeds)

# 不同iable augmentation - Karras et al. stylegan-ada
# 从水平翻转开始

# 定义类 DiffAugment，继承自 nn.Module
class DiffAugment(nn.Module):
    # 初始化方法，接受概率 prob、是否进行水平翻转 horizontal_flip 和水平翻转概率 horizontal_flip_prob
    def __init__(
        self,
        *,
        prob,
        horizontal_flip,
        horizontal_flip_prob = 0.5
    ):
        super().__init__()
        self.prob = prob
        assert 0 <= prob <= 1.

        self.horizontal_flip = horizontal_flip
        self.horizontal_flip_prob = horizontal_flip_prob

    # 前向传播方法，接受 images 和 rgbs
    def forward(
        self,
        images,
        rgbs: List[Tensor]
    ):
        # 如果随机数大于等于概率 prob，则直接返回 images 和 rgbs
        if random() >= self.prob:
            return images, rgbs

        # 如果随机数小于水平翻转概率 horizontal_flip_prob，则对 images 和 rgbs 进行水平翻转
        if random() < self.horizontal_flip_prob:
            images = torch.flip(images, (-1,))
            rgbs = [torch.flip(rgb, (-1,)) for rgb in rgbs]

        return images, rgbs

# rmsnorm（新论文显示在 layernorm 中进行均值中心化不是必要的）

# 定义类 ChannelRMSNorm，继承自 nn.Module
class ChannelRMSNorm(nn.Module):
    # 初始化方法，接受维度 dim
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    # 前向传播方法，对输入 x 进行归一化处���并乘以缩放因子和 gamma 参数
    def forward(self, x):
        normed = F.normalize(x, dim = 1)
        return normed * self.scale * self.gamma

# 定义类 RMSNorm，继承自 nn.Module
class RMSNorm(nn.Module):
    # 初始化方法，接受维度 dim
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    # 前向传播方法，对输入 x 进行归一化处理并乘以缩放因子和 gamma 参数
    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

# 下采样和上采样

# 定义类 Blur，继承自 nn.Module
class Blur(nn.Module):
    # 初始化方法，创建一个张量 f，并注册为缓冲区
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    # 前向传播方法，对输入 x 进行二维卷积滤波
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized = True)

# 定义��数 Upsample，返回一个包含上采样和模糊处理的序列
def Upsample(*args):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False),
        Blur()
    )

# 定义类 PixelShuffleUpsample，继承自 nn.Module
class PixelShuffleUpsample(nn.Module):
    # 初始化方法，接受维度 dim
    def __init__(self, dim):
        super().__init__()
        conv = nn.Conv2d(dim, dim * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    # 初始化卷积层的权重
    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    # 前向传播方法，对输入 x 进行处理
    def forward(self, x):
        return self.net(x)

# 定义函数 Downsample，返回一个包含下采样和卷积层的序列
def Downsample(dim):
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim, 1)
    )

# 跳跃层激励

# 定义函数 SqueezeExcite，返回一个包含减少、线性层、SiLU 激活函数、线性层、Sigmoid 激活函数和重排维度的序列
def SqueezeExcite(dim, dim_out, reduction = 4, dim_min = 32):
    dim_hidden = max(dim_out // reduction, dim_min)

    return nn.Sequential(
        Reduce('b c h w -> b c', 'mean'),
        nn.Linear(dim, dim_hidden),
        nn.SiLU(),
        nn.Linear(dim_hidden, dim_out),
        nn.Sigmoid(),
        Rearrange('b c -> b c 1 1')
    )

# 自适应卷积
# 论文的主要创新 - 他们提出根据文本嵌入学习 N 个卷积核的 softmax 加权和

# 定义函数 get_same_padding，计算卷积层的 padding 大小
def get_same_padding(size, kernel, dilation, stride):
    return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

# 定义类 AdaptiveConv2DMod，继承自 nn.Module
class AdaptiveConv2DMod(nn.Module):
    # 初始化函数，设置卷积层的参数
    def __init__(
        self,
        dim,
        dim_out,
        kernel,
        *,
        demod = True,
        stride = 1,
        dilation = 1,
        eps = 1e-8,
        num_conv_kernels = 1 # set this to be greater than 1 for adaptive
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置 epsilon 值
        self.eps = eps

        # 设置输出维度
        self.dim_out = dim_out

        # 设置卷积核大小、步长、膨胀率
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        # 是否使用自适应卷积核
        self.adaptive = num_conv_kernels > 1

        # 初始化权重参数
        self.weights = nn.Parameter(torch.randn((num_conv_kernels, dim_out, dim, kernel, kernel)))

        # 是否使用 demodulation
        self.demod = demod

        # 使用 kaiming_normal 初始化权重
        nn.init.kaiming_normal_(self.weights, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    # 前向传播函数
    def forward(
        self,
        fmap,
        mod: Optional[Tensor] = None,
        kernel_mod: Optional[Tensor] = None
    ):
        """
        notation

        b - batch
        n - convs
        o - output
        i - input
        k - kernel
        """

        # 获取 batch 大小和特征图高度
        b, h = fmap.shape[0], fmap.shape[-2]

        # 考虑特征图在第一维度上由于多尺度输入和输出而扩展的情况

        # 如果 mod 的 batch 大小不等于 b，则进行重复操作
        if mod.shape[0] != b:
            mod = repeat(mod, 'b ... -> (s b) ...', s = b // mod.shape[0])

        # 如果存在 kernel_mod
        if exists(kernel_mod):
            kernel_mod_has_el = kernel_mod.numel() > 0

            # 如果使用自适应卷积核，kernel_mod 必须为空
            assert self.adaptive or not kernel_mod_has_el

            # 如果 kernel_mod 不为空且其 batch 大小不等于 b，则进行重复操作
            if kernel_mod_has_el and kernel_mod.shape[0] != b:
                kernel_mod = repeat(kernel_mod, 'b ... -> (s b) ...', s = b // kernel_mod.shape[0])

        # 准备用于调制的权重

        weights = self.weights

        # 如果使用自适应卷积核
        if self.adaptive:
            # 对权重进行重复操作
            weights = repeat(weights, '... -> b ...', b = b)

            # 确定自适应权重并使用 softmax 选择要使用的卷积核
            assert exists(kernel_mod) and kernel_mod.numel() > 0

            kernel_attn = kernel_mod.softmax(dim = -1)
            kernel_attn = rearrange(kernel_attn, 'b n -> b n 1 1 1 1')

            weights = reduce(weights * kernel_attn, 'b n ... -> b ...', 'sum')

        # 进行调制和解调制，类似 stylegan2 中的操作

        mod = rearrange(mod, 'b i -> b 1 i 1 1')

        weights = weights * (mod + 1)

        # 如果使用解调制
        if self.demod:
            inv_norm = reduce(weights ** 2, 'b o i k1 k2 -> b o 1 1 1', 'sum').clamp(min = self.eps).rsqrt()
            weights = weights * inv_norm

        fmap = rearrange(fmap, 'b c h w -> 1 (b c) h w')

        weights = rearrange(weights, 'b o ... -> (b o) ...')

        # 计算填充值
        padding = get_same_padding(h, self.kernel, self.dilation, self.stride)
        # 使用卷积操作
        fmap = F.conv2d(fmap, weights, padding = padding, groups = b)

        # 重新排列特征图
        return rearrange(fmap, '1 (b o) ... -> b o ...', b = b)
# 定义 SelfAttention 类，用于实现自注意力机制
class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dot_product = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.dot_product = dot_product

        self.norm = ChannelRMSNorm(dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_inner, 1, bias = False) if dot_product else None
        self.to_v = nn.Conv2d(dim, dim_inner, 1, bias = False)

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))

        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)

    # 实现前向传播函数
    def forward(self, fmap):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = fmap.shape[0]

        fmap = self.norm(fmap)

        x, y = fmap.shape[-2:]

        h = self.heads

        q, v = self.to_q(fmap), self.to_v(fmap)

        k = self.to_k(fmap) if exists(self.to_k) else q

        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = self.heads), (q, k, v))

        # add a null key / value, so network can choose to pay attention to nothing

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b = batch), self.null_kv)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # l2 distance or dot product

        if self.dot_product:
            sim = einsum('b i d, b j d -> b i j', q, k)
        else:
            # using pytorch cdist leads to nans in lightweight gan training framework, at least
            q_squared = (q * q).sum(dim = -1)
            k_squared = (k * k).sum(dim = -1)
            l2dist_squared = rearrange(q_squared, 'b i -> b i 1') + rearrange(k_squared, 'b j -> b 1 j') - 2 * einsum('b i d, b j d -> b i j', q, k) # hope i'm mathing right
            sim = -l2dist_squared

        # scale

        sim = sim * self.scale

        # attention

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)

        return self.to_out(out)

# 定义 CrossAttention 类，用于实现交叉注意力机制
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        kv_input_dim = default(dim_context, dim)

        self.norm = ChannelRMSNorm(dim)
        self.norm_context = RMSNorm(kv_input_dim)

        self.to_q = nn.Conv2d(dim, dim_inner, 1, bias = False)
        self.to_kv = nn.Linear(kv_input_dim, dim_inner * 2, bias = False)
        self.to_out = nn.Conv2d(dim_inner, dim, 1, bias = False)
    # 定义一个前向传播函数，接受特征图、上下文和可选的掩码作为输入
    def forward(self, fmap, context, mask = None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """

        # 对特征图进行归一化处理
        fmap = self.norm(fmap)
        # 对上下文进行归一化处理
        context = self.norm_context(context)

        # 获取特征图的高度和宽度
        x, y = fmap.shape[-2:]

        # 获取头数
        h = self.heads

        # 将特征图转换为查询、键、值
        q, k, v = (self.to_q(fmap), *self.to_kv(context).chunk(2, dim = -1))

        # 将键和值重排维度
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (k, v))

        # 重排查询维度
        q = rearrange(q, 'b (h d) x y -> (b h) (x y) d', h = self.heads)

        # 计算查询和键之间的相似度
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 如果存在掩码，则进行掩码处理
        if exists(mask):
            mask = repeat(mask, 'b j -> (b h) 1 j', h = self.heads)
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 对相似度进行 softmax 操作得到注意力权重
        attn = sim.softmax(dim = -1)

        # 根据注意力权重计算输出
        out = einsum('b i j, b j d -> b i d', attn, v)

        # 重排输出维度
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', x = x, y = y, h = h)

        # 将输出转换为最终输出
        return self.to_out(out)
# 定义经典的 transformer 注意力机制，使用 L2 距离

class TextAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)  # 初始化 RMS 归一化层
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)  # 初始化线性层，用于计算查询、键、值

        self.null_kv = nn.Parameter(torch.randn(2, heads, dim_head))  # 初始化空键/值参数

        self.to_out = nn.Linear(dim_inner, dim, bias = False)  # 初始化输出线性层

    def forward(self, encodings, mask = None):
        """
        einstein notation

        b - batch
        h - heads
        x - height
        y - width
        d - dimension
        i - source seq (attend from)
        j - target seq (attend to)
        """
        batch = encodings.shape[0]

        encodings = self.norm(encodings)  # 对输入进行归一化处理

        h = self.heads

        q, k, v = self.to_qkv(encodings).chunk(3, dim = -1)  # 将查询、键、值分割
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))  # 重排形状

        # 添加一个空键/值，以便网络可以选择不关注任何内容

        nk, nv = map(lambda t: repeat(t, 'h d -> (b h) 1 d', b = batch), self.null_kv)  # 重复空键/值

        k = torch.cat((nk, k), dim = -2)  # 拼接键
        v = torch.cat((nv, v), dim = -2)  # 拼接值

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale  # 计算相似度

        # 键填充掩码

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)  # 对掩码进行填充
            mask = repeat(mask, 'b n -> (b h) 1 n', h = h)  # 重复掩码
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)  # 对掩码外的值进行替换

        # 注意力

        attn = sim.softmax(dim = -1)  # 计算注意力权重
        out = einsum('b i j, b j d -> b i d', attn, v)  # 计算输出

        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)  # 重排输出形状

        return self.to_out(out)  # 返回输出结果

# 前馈网络

def FeedForward(
    dim,
    mult = 4,
    channel_first = False
):
    dim_hidden = int(dim * mult)
    norm_klass = ChannelRMSNorm if channel_first else RMSNorm
    proj = partial(nn.Conv2d, kernel_size = 1) if channel_first else nn.Linear

    return nn.Sequential(
        norm_klass(dim),  # 初始化归一化层
        proj(dim, dim_hidden),  # 线性变换到隐藏维度
        nn.GELU(),  # GELU 激活函数
        proj(dim_hidden, dim)  # 线性变换回原始维度
    )

# 不同类型的 transformer 块或 transformer（多个块）

class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        dot_product = False
    ):
        super().__init__()
        self.attn = SelfAttention(dim = dim, dim_head = dim_head, heads = heads, dot_product = dot_product)  # 初始化自注意力层
        self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)  # 初始化前馈网络

    def forward(self, x):
        x = self.attn(x) + x  # 自注意力操作后加上残差连接
        x = self.ff(x) + x  # 前馈网络操作后加上残差连接
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_context,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.attn = CrossAttention(dim = dim, dim_context = dim_context, dim_head = dim_head, heads = heads)  # 初始化交叉注意力层
        self.ff = FeedForward(dim = dim, mult = ff_mult, channel_first = True)  # 初始化前馈网络

    def forward(self, x, context, mask = None):
        x = self.attn(x, context = context, mask = mask) + x  # 交叉注意力操作后加上残差连接
        x = self.ff(x) + x  # 前馈网络操作后加上残差连接
        return x

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TextAttention(dim = dim, dim_head = dim_head, heads = heads),  # 添加文本注意力层
                FeedForward(dim = dim, mult = ff_mult)  # 添加前馈网络
            ]))

        self.norm = RMSNorm(dim)  # 初始化 RMS 归一化层
    # 定义前向传播函数，接受输入 x 和掩码 mask，默认为 None
    def forward(self, x, mask = None):
        # 遍历每个注意力层和前馈神经网络层
        for attn, ff in self.layers:
            # 使用注意力层处理输入 x，并将结果与 x 相加
            x = attn(x, mask = mask) + x
            # 使用前馈神经网络层处理输入 x，并将结果与 x 相加
            x = ff(x) + x

        # 对处理后的 x 进行归一化处理
        return self.norm(x)
# 文本编码器类
class TextEncoder(nn.Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        *,
        dim,
        depth,
        clip: Optional[OpenClipAdapter] = None,
        dim_head = 64,
        heads = 8,
    ):
        super().__init__()
        self.dim = dim

        # 如果 clip 不存在，则创建一个 OpenClipAdapter 对象
        if not exists(clip):
            clip = OpenClipAdapter()

        self.clip = clip
        # 设置 clip 不需要梯度
        set_requires_grad_(clip, False)

        # 创建一个学习到的全局标记
        self.learned_global_token = nn.Parameter(torch.randn(dim))

        # 根据 clip.dim_latent 是否等于 dim 来选择 Linear 层或 Identity 函数
        self.project_in = nn.Linear(clip.dim_latent, dim) if clip.dim_latent != dim else nn.Identity()

        # 创建一个 Transformer 模型
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads
        )

    # 前向传播函数
    @beartype
    def forward(
        self,
        texts: Optional[List[str]] = None,
        text_encodings: Optional[Tensor] = None
    ):
        # texts 和 text_encodings 必须有且只有一个存在
        assert exists(texts) ^ exists(text_encodings)

        # 如果 text_encodings 不存在，则使用 texts 通过 clip.embed_texts 方法获取
        if not exists(text_encodings):
            with torch.no_grad():
                self.clip.eval()
                _, text_encodings = self.clip.embed_texts(texts)

        # 创建一个 mask，用于标记 text_encodings 中不为 0 的位置
        mask = (text_encodings != 0.).any(dim = -1)

        # 对 text_encodings 进行线性变换
        text_encodings = self.project_in(text_encodings)

        # 在 mask 前面填充一个 True 值，用于表示全局标记
        mask_with_global = F.pad(mask, (1, 0), value = True)

        # 获取 batch 大小，并重复学习到的全局标记
        batch = text_encodings.shape[0]
        global_tokens = repeat(self.learned_global_token, 'd -> b d', b = batch)

        # 打包全局标记和 text_encodings
        text_encodings, ps = pack([global_tokens, text_encodings], 'b * d')

        # 使用 Transformer 模型进行编码
        text_encodings = self.transformer(text_encodings, mask = mask_with_global)

        # 解包结果，获取全局标记和编码结果
        global_tokens, text_encodings = unpack(text_encodings, ps, 'b * d')

        # 返回全局标记��编码结果和 mask
        return global_tokens, text_encodings, mask

# 等权线性层
class EqualLinear(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        dim_out,
        lr_mul = 1,
        bias = True
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim_out, dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim_out))

        self.lr_mul = lr_mul

    # 前向传播函数
    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

# 风格网络类
class StyleNetwork(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        depth,
        lr_mul = 0.1,
        dim_text_latent = 0
    ):
        super().__init__()
        self.dim = dim
        self.dim_text_latent = dim_text_latent

        layers = []
        # 构建深度为 depth 的网络层
        for i in range(depth):
            is_first = i == 0
            dim_in = (dim + dim_text_latent) if is_first else dim

            layers.extend([EqualLinear(dim_in, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    # 前向传播函数
    def forward(
        self,
        x,
        text_latent = None
    ):
        # 对输入 x 进行归一化
        x = F.normalize(x, dim = 1)

        # 如果 dim_text_latent 大于 0，则将 text_latent 拼接到 x 中
        if self.dim_text_latent > 0:
            assert exists(text_latent)
            x = torch.cat((x, text_latent), dim = -1)

        # 返回网络处理后的结果
        return self.net(x)

# 噪声类
class Noise(nn.Module):
    # 初始化函数
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim, 1, 1))

    # 前向传播函数
    def forward(
        self,
        x,
        noise = None
    ):
        b, _, h, w, device = *x.shape, x.device

        # 如果 noise 不存在，则创建一个随机噪声
        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device = device)

        # 返回加上噪声的结果
        return x + self.weight * noise

# 生成器基类
class BaseGenerator(nn.Module):
    pass

# 生成器类
class Generator(BaseGenerator):
    # 初始化函数
    @beartype
    # 初始化函数，设置模型参数
    def __init__(
        self,
        *,
        image_size,  # 图像尺寸
        dim_capacity = 16,  # 容量维度
        dim_max = 2048,  # 最大维度
        channels = 3,  # 通道数
        style_network: Optional[Union[StyleNetwork, Dict]] = None,  # 风格网络
        style_network_dim = None,  # 风格网络维度
        text_encoder: Optional[Union[TextEncoder, Dict]] = None,  # 文本编码器
        dim_latent = 512,  # 潜在维度
        self_attn_resolutions: Tuple[int, ...] = (32, 16),  # 自注意力分辨率
        self_attn_dim_head = 64,  # 自注意力头维度
        self_attn_heads = 8,  # 自注意力头数
        self_attn_dot_product = True,  # 自注意力是否使用点积
        self_attn_ff_mult = 4,  # 自注意力前馈倍数
        cross_attn_resolutions: Tuple[int, ...] = (32, 16),  # 交叉注意力分辨率
        cross_attn_dim_head = 64,  # 交叉注意力头维度
        cross_attn_heads = 8,  # 交叉注意力头数
        cross_attn_ff_mult = 4,  # 交叉注意力前馈倍数
        num_conv_kernels = 2,  # 自适应卷积核数量
        num_skip_layers_excite = 0,  # 激励跳层数量
        unconditional = False,  # 是否无条件
        pixel_shuffle_upsample = False  # 像素混洗上采样
    def init_(self, m):
        # 初始化函数，使用 kaiming_normal 初始化卷积和全连接层的权重
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    @property
    def total_params(self):
        # 计算模型总参数数量
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    @property
    def device(self):
        # 获取模型所在设备
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        styles = None,  # 风格
        noise = None,  # 噪声
        texts: Optional[List[str]] = None,  # 文本列表
        text_encodings: Optional[Tensor] = None,  # 文本编码
        global_text_tokens = None,  # 全局文本标记
        fine_text_tokens = None,  # 精细文本标记
        text_mask = None,  # 文本掩码
        batch_size = 1,  # 批量大小
        return_all_rgbs = False  # 是否返回所有 RGB
        ):
            # 处理文本编码
            # 需要全局文本令牌来自适应选择主要贡献中的内核
            # 需要细文本令牌来使用交叉注意力

            if not self.unconditional:
                if exists(texts) or exists(text_encodings):
                    assert exists(texts) ^ exists(text_encodings), '要么传入原始文本作为 List[str]，要么传入文本编码（来自 clip）作为 Tensor，但不能同时传入'
                    assert exists(self.text_encoder)

                    if exists(texts):
                        text_encoder_kwargs = dict(texts = texts)
                    elif exists(text_encodings):
                        text_encoder_kwargs = dict(text_encodings = text_encodings)

                    global_text_tokens, fine_text_tokens, text_mask = self.text_encoder(**text_encoder_kwargs)
                else:
                    assert all([*map(exists, (global_text_tokens, fine_text_tokens, text_mask)]), '未传入原始文本或文本嵌入以进行条件训练'
            else:
                assert not any([*map(exists, (texts, global_text_tokens, fine_text_tokens))])

            # 确定风格

            if not exists(styles):
                assert exists(self.style_network)

                if not exists(noise):
                    noise = torch.randn((batch_size, self.style_network_dim), device = self.device)

                styles = self.style_network(noise, global_text_tokens)

            # 将风格投影到卷积调制

            conv_mods = self.style_to_conv_modulations(styles)
            conv_mods = conv_mods.split(self.style_embed_split_dims, dim = -1)
            conv_mods = iter(conv_mods)

            # 准备初始块

            batch_size = styles.shape[0]

            x = repeat(self.init_block, 'c h w -> b c h w', b = batch_size)
            x = self.init_conv(x, mod = next(conv_mods), kernel_mod = next(conv_mods))

            rgb = torch.zeros((batch_size, self.channels, 4, 4), device = self.device, dtype = x.dtype)

            # 跳过层挤压激发

            excitations = [None] * self.num_skip_layers_excite

            # 保存生成器每一层的所有 rgb 用于多分辨率输入判别

            rgbs = []

            # 主网络

            for squeeze_excite, (resnet_conv1, noise1, act1, resnet_conv2, noise2, act2), to_rgb_conv, self_attn, cross_attn, upsample, upsample_rgb in self.layers:

                if exists(upsample):
                    x = upsample(x)

                if exists(squeeze_excite):
                    skip_excite = squeeze_excite(x)
                    excitations.append(skip_excite)

                excite = safe_unshift(excitations)
                if exists(excite):
                    x = x * excite

                x = resnet_conv1(x, mod = next(conv_mods), kernel_mod = next(conv_mods))
                x = noise1(x)
                x = act1(x)

                x = resnet_conv2(x, mod = next(conv_mods), kernel_mod = next(conv_mods))
                x = noise2(x)
                x = act2(x)

                if exists(self_attn):
                    x = self_attn(x)

                if exists(cross_attn):
                    x = cross_attn(x, context = fine_text_tokens, mask = text_mask)

                layer_rgb = to_rgb_conv(x, mod = next(conv_mods), kernel_mod = next(conv_mods))

                rgb = rgb + layer_rgb

                rgbs.append(rgb)

                if exists(upsample_rgb):
                    rgb = upsample_rgb(rgb)

            # 检查

            assert is_empty([*conv_mods]), '卷积错误调制'

            if return_all_rgbs:
                return rgb, rgbs

            return rgb
# 定义一个简单的解码器类，继承自 nn.Module
@beartype
class SimpleDecoder(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        *,
        dims: Tuple[int, ...],
        patch_dim: int = 1,
        frac_patches: float = 1.,
        dropout: float = 0.5
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 断言确保 frac_patches 在 0 到 1 之间
        assert 0 < frac_patches <= 1.

        # 初始化一些参数
        self.patch_dim = patch_dim
        self.frac_patches = frac_patches

        # 创建一个 dropout 层
        self.dropout = nn.Dropout(dropout)

        # 将 dim 和 dims 组成一个列表
        dims = [dim, *dims]

        # 初始化一个空的层列表
        layers = [conv2d_3x3(dim, dim)]

        # 遍历 dims 列表，创建卷积层和激活函数层
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Sequential(
                Upsample(dim_in),
                conv2d_3x3(dim_in, dim_out),
                leaky_relu()
            ))

        # 创建一个包含所有层的神经网络
        self.net = nn.Sequential(*layers)

    # 定义一个属性，返回参数的设备信息
    @property
    def device(self):
        return next(self.parameters()).device

    # 前向传播函数，接受特征图和原始图像作为输入
    def forward(
        self,
        fmap,
        orig_image
    ):
        # 对特征图进行 dropout
        fmap = self.dropout(fmap)

        # 如果 frac_patches 小于 1
        if self.frac_patches < 1.:
            # 获取 batch 大小和 patch 维度
            batch, patch_dim = fmap.shape[0], self.patch_dim
            fmap_size, img_size = fmap.shape[-1], orig_image.shape[-1]

            # 断言确保特征图和图像大小能够整除 patch 维度
            assert divisible_by(fmap_size, patch_dim), f'feature map dimensions are {fmap_size}, but the patch dim was designated to be {patch_dim}'
            assert divisible_by(img_size, patch_dim), f'image size is {img_size} but the patch dim was specified to be {patch_dim}'

            # 重排特征图和原始图像的维度
            fmap, orig_image = map(lambda t: rearrange(t, 'b c (p1 h) (p2 w) -> b (p1 p2) c h w', p1 = patch_dim, p2 = patch_dim), (fmap, orig_image))

            # 计算总 patch 数量和需要重建的 patch 数量
            total_patches = patch_dim ** 2
            num_patches_recon = max(int(self.frac_patches * total_patches), 1)

            # 创建一个 batch 的索引和随机排列的索引
            batch_arange = torch.arange(batch, device = self.device)[..., None]
            batch_randperm = torch.randn((batch, total_patches)).sort(dim = -1).indices
            patch_indices = batch_randperm[..., :num_patches_recon]

            # 从特征图和原始图像中选择对应的 patch
            fmap, orig_image = map(lambda t: t[batch_arange, patch_indices], (fmap, orig_image))
            fmap, orig_image = map(lambda t: rearrange(t, 'b p ... -> (b p) ...'), (fmap, orig_image))

        # 将选定的 patch 输入神经网络进行重建
        recon = self.net(fmap)
        # 返回重建图像和原始图像的均方误差损失
        return F.mse_loss(recon, orig_image)

# 定义一个随机固定投影类，继承自 nn.Module
class RandomFixedProjection(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        dim_out,
        channel_first = True
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 生成随机权重并初始化
        weights = torch.randn(dim, dim_out)
        nn.init.kaiming_normal_(weights, mode = 'fan_out', nonlinearity = 'linear')

        # 初始化一些参数
        self.channel_first = channel_first
        self.register_buffer('fixed_weights', weights)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 如果 channel_first 为 False，则返回 x 与固定权重的矩阵乘积
        if not self.channel_first:
            return x @ self.fixed_weights

        # 如果 channel_first 为 True，则返回 x 与固定权重的张量乘积
        return einsum('b c ..., c d -> b d ...', x, self.fixed_weights)

# 定义一个视觉辅助鉴别器类，继承自 nn.Module
class VisionAidedDiscriminator(nn.Module):
    """ the vision-aided gan loss """

    # 初始化函数，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        depth = 2,
        dim_head = 64,
        heads = 8,
        clip: Optional[OpenClipAdapter] = None,
        layer_indices = (-1, -2, -3),
        conv_dim = None,
        text_dim = None,
        unconditional = False,
        num_conv_kernels = 2
    ):
        # 调用父类的构造函数
        super().__init__()

        # 如果指定的 clip 不存在，则使用 OpenClipAdapter() 创建一个 clip 对象
        if not exists(clip):
            clip = OpenClipAdapter()

        # 设置对象的 clip 属性为传入的 clip 参数
        self.clip = clip
        # 获取 clip 对象的 _dim_image_latent 属性值
        dim = clip._dim_image_latent

        # 设置 unconditional 属性为传入的 unconditional 参数，如果未传入则使用 dim 的值
        self.unconditional = unconditional
        text_dim = default(text_dim, dim)
        conv_dim = default(conv_dim, dim)

        # 初始化 layer_discriminators 为一个空的 nn.ModuleList
        self.layer_discriminators = nn.ModuleList([])
        # 设置 layer_indices 属性为传入的 layer_indices 参数

        # 根据 unconditional 的值选择不同的卷积类
        conv_klass = partial(AdaptiveConv2DMod, kernel = 3, num_conv_kernels = num_conv_kernels) if not unconditional else conv2d_3x3

        # 遍历 layer_indices，为每个索引创建一个包含不同模块的 nn.ModuleList
        for _ in layer_indices:
            self.layer_discriminators.append(nn.ModuleList([
                RandomFixedProjection(dim, conv_dim),
                conv_klass(conv_dim, conv_dim),
                nn.Linear(text_dim, conv_dim) if not unconditional else None,
                nn.Linear(text_dim, num_conv_kernels) if not unconditional else None,
                nn.Sequential(
                    conv2d_3x3(conv_dim, 1),
                    Rearrange('b 1 ... -> b ...')
                )
            ]))

    # 返回 layer_discriminators 中所有模块的参数
    def parameters(self):
        return self.layer_discriminators.parameters()

    # 返回 layer_discriminators 中所有模块参数的总数量
    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    # 前向传播函数，接收 images、texts、text_embeds 和 return_clip_encodings 参数
    @beartype
    def forward(
        self,
        images,
        texts: Optional[List[str]] = None,
        text_embeds: Optional[Tensor] = None,
        return_clip_encodings = False
    ):

        # 断言条件，确保在有条件生成时存在 text_embeds 或 texts
        assert self.unconditional or (exists(text_embeds) ^ exists(texts))

        # 在无条件生成且存在 texts 时，使用 clip 对象的 embed_texts 属性作为 text_embeds
        with torch.no_grad():
            if not self.unconditional and exists(texts):
                self.clip.eval()
                text_embeds = self.clip.embed_texts

        # 获取 images 的编码结果
        _, image_encodings = self.clip.embed_images(images)

        # 初始化 logits 列表
        logits = []

        # 遍历 layer_indices 和 layer_discriminators 中的模块，计算 logits
        for layer_index, (rand_proj, conv, to_conv_mod, to_conv_kernel_mod, to_logits) in zip(self.layer_indices, self.layer_discriminators):
            image_encoding = image_encodings[layer_index]

            cls_token, rest_tokens = image_encoding[:, :1], image_encoding[:, 1:]
            height_width = int(sqrt(rest_tokens.shape[-2])) # 假设为正方形

            img_fmap = rearrange(rest_tokens, 'b (h w) d -> b d h w', h = height_width)

            img_fmap = img_fmap + rearrange(cls_token, 'b 1 d -> b d 1 1 ') # 将 cls token 汇入其余 token

            img_fmap = rand_proj(img_fmap)

            if self.unconditional:
                img_fmap = conv(img_fmap)
            else:
                assert exists(text_embeds)

                img_fmap = conv(
                    img_fmap,
                    mod = to_conv_mod(text_embeds),
                    kernel_mod = to_conv_kernel_mod(text_embeds)
                )

            layer_logits = to_logits(img_fmap)

            logits.append(layer_logits)

        # 如果不需要返回 clip 编码，则返回 logits
        if not return_clip_encodings:
            return logits

        # 否则返回 logits 和 image_encodings
        return logits, image_encodings
# 定义一个预测器类，继承自 nn.Module
class Predictor(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        depth = 4,
        num_conv_kernels = 2,
        unconditional = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置是否无条件的标志
        self.unconditional = unconditional
        # 创建一个卷积层，用于残差连接
        self.residual_fn = nn.Conv2d(dim, dim, 1)
        # 设置残差缩放因子
        self.residual_scale = 2 ** -0.5

        # 创建一个空的模块列表
        self.layers = nn.ModuleList([])

        # 根据是否无条件，选择不同的卷积类
        klass = nn.Conv2d if unconditional else partial(AdaptiveConv2DMod, num_conv_kernels = num_conv_kernels)
        klass_kwargs = dict(padding = 1) if unconditional else dict()

        # 循环创建深度次数的卷积层
        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                klass(dim, dim, 3, **klass_kwargs),
                leaky_relu(),
                klass(dim, dim, 3, **klass_kwargs),
                leaky_relu()
            ]))

        # 创建一个转换为 logits 的卷积层
        self.to_logits = nn.Conv2d(dim, 1, 1)

    # 前向传播函数，接受多个参数
    def forward(
        self,
        x,
        mod = None,
        kernel_mod = None
    ):
        # 计算残差
        residual = self.residual_fn(x)

        kwargs = dict()

        # 如果不是无条件的，则传入 mod 和 kernel_mod 参数
        if not self.unconditional:
            kwargs = dict(mod = mod, kernel_mod = kernel_mod)

        # 循环处理每一层
        for conv1, activation, conv2, activation in self.layers:

            inner_residual = x

            x = conv1(x, **kwargs)
            x = activation(x)
            x = conv2(x, **kwargs)
            x = activation(x)

            x = x + inner_residual
            x = x * self.residual_scale

        # 加上残差并返回 logits
        x = x + residual
        return self.to_logits(x)

# 定义一个鉴别器类，继承自 nn.Module
class Discriminator(nn.Module):
    # 初始化函数，接受多个参数
    @beartype
    def __init__(
        self,
        *,
        dim_capacity = 16,
        image_size,
        dim_max = 2048,
        channels = 3,
        attn_resolutions: Tuple[int, ...] = (32, 16),
        attn_dim_head = 64,
        attn_heads = 8,
        self_attn_dot_product = False,
        ff_mult = 4,
        text_encoder: Optional[Union[TextEncoder, Dict]] = None,
        text_dim = None,
        filter_input_resolutions: bool = True,
        multiscale_input_resolutions: Tuple[int, ...] = (64, 32, 16, 8),
        multiscale_output_skip_stages: int = 1,
        aux_recon_resolutions: Tuple[int, ...] = (8,),
        aux_recon_patch_dims: Tuple[int, ...] = (2,),
        aux_recon_frac_patches: Tuple[float, ...] = (0.25,),
        aux_recon_fmap_dropout: float = 0.5,
        resize_mode = 'bilinear',
        num_conv_kernels = 2,
        num_skip_layers_excite = 0,
        unconditional = False,
        predictor_depth = 2
    def init_(self, m):
        # 初始化函数，对卷积和线性层进行初始化
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'leaky_relu')

    # 将图像调整大小到指定分辨率
    def resize_image_to(self, images, resolution):
        return F.interpolate(images, resolution, mode = self.resize_mode)

    # 将真实图像调整大小到多个分辨率
    def real_images_to_rgbs(self, images):
        return [self.resize_image_to(images, resolution) for resolution in self.multiscale_input_resolutions]

    # 返回模型的总参数数量
    @property
    def total_params(self):
        return sum([p.numel() for p in self.parameters()])

    # 返回模型所在设备
    @property
    def device(self):
        return next(self.parameters()).device

    # 前向传播函数，接受多个参数
    @beartype
    def forward(
        self,
        images,
        rgbs: List[Tensor],                   # 生成器的多分辨率输入
        texts: Optional[List[str]] = None,
        text_encodings: Optional[Tensor] = None,
        text_embeds = None,
        real_images = None,                   # 如果传入真实图像，网络将自动将其附加到传入的生成图像中，并通过适当的调整大小和连接生成所有中间分辨率
        return_multiscale_outputs = True,     # 可以强制不返回多尺度 logits
        calc_aux_loss = True
# gan

# 定义训练鉴别器损失的命名元组
TrainDiscrLosses = namedtuple('TrainDiscrLosses', [
    'divergence',
    'multiscale_divergence',
    'vision_aided_divergence',
    'total_matching_aware_loss',
    'gradient_penalty',
    'aux_reconstruction'
])
# 定义一个命名元组，包含训练生成器的损失值
TrainGenLosses = namedtuple('TrainGenLosses', [
    'divergence',
    'multiscale_divergence',
    'total_vd_divergence',
    'contrastive_loss'
])

# 定义 GigaGAN 类，继承自 nn.Module
class GigaGAN(nn.Module):
    # 初始化函数
    @beartype
    def __init__(
        self,
        *,
        generator: Union[BaseGenerator, Dict],  # 生成器对象或字典
        discriminator: Union[Discriminator, Dict],  # 判别器对象或字典
        vision_aided_discriminator: Optional[Union[VisionAidedDiscriminator, Dict]] = None,  # 辅助视觉判别器对象或字典
        diff_augment: Optional[Union[DiffAugment, Dict]] = None,  # 数据增强对象或字典
        learning_rate = 2e-4,  # 学习率
        betas = (0.5, 0.9),  # Adam 优化器的 beta 参数
        weight_decay = 0.,  # 权重衰减
        discr_aux_recon_loss_weight = 1.,  # 判别器辅助重建损失权重
        multiscale_divergence_loss_weight = 0.1,  # 多尺度散度损失权重
        vision_aided_divergence_loss_weight = 0.5,  # 视觉辅助散度损失权重
        generator_contrastive_loss_weight = 0.1,  # 生成器对比损失权重
        matching_awareness_loss_weight = 0.1,  # 匹配感知损失权重
        calc_multiscale_loss_every = 1,  # 计算多尺度损失的频率
        apply_gradient_penalty_every = 4,  # 应用梯度惩罚的频率
        resize_image_mode = 'bilinear',  # 图像调整模式
        train_upsampler = False,  # 是否训练上采样器
        log_steps_every = 20,  # 每隔多少步记录日志
        create_ema_generator_at_init = True,  # 是否在初始化时创建 EMA 生成器
        save_and_sample_every = 1000,  # 保存和采样的频率
        early_save_thres_steps = 2500,  # 早期保存的阈值步数
        early_save_and_sample_every = 100,  # 早期保存和采样的频率
        num_samples = 25,  # 采样数量
        model_folder = './gigagan-models',  # 模型保存文件夹路径
        results_folder = './gigagan-results',  # 结果保存文件夹路径
        sample_upsampler_dl: Optional[DataLoader] = None,  # 上采样器数据加载器
        accelerator: Optional[Accelerator] = None,  # 加速器对象
        accelerate_kwargs: dict = {},  # 加速参数
        find_unused_parameters = True,  # 是否查找未使用的参数
        amp = False,  # 是否使用混合精度训练
        mixed_precision_type = 'fp16'  # 混合精度类型
    # 保存模型的方法
    def save(self, path, overwrite = True):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 如果父目录不存在，则创建
        mkdir_if_not_exists(path.parents[0])

        # 断言是否覆盖保存或路径不存在
        assert overwrite or not path.exists()

        # 创建包含模型参数的字典
        pkg = dict(
            G = self.unwrapped_G.state_dict(),  # 生成器的状态字典
            D = self.unwrapped_D.state_dict(),  # 判别器的状态字典
            G_opt = self.G_opt.state_dict(),  # 生成器优化器的状态字典
            D_opt = self.D_opt.state_dict(),  # 判别器优化器的状态字典
            steps = self.steps.item(),  # 训练步数
            version = __version__  # 版本号
        )

        # 如果存在生成器优化器的 scaler，则保存其状态字典
        if exists(self.G_opt.scaler):
            pkg['G_scaler'] = self.G_opt.scaler.state_dict()

        # 如果存在判别器优化器的 scaler，则保存其状态字典
        if exists(self.D_opt.scaler):
            pkg['D_scaler'] = self.D_opt.scaler.state_dict()

        # 如果存在视觉辅助判别器，则保存其状态字典
        if exists(self.VD):
            pkg['VD'] = self.unwrapped_VD.state_dict()
            pkg['VD_opt'] = self.VD_opt.state_dict()

            # 如果存在视觉辅助判别器的 scaler，则保存其状态字典
            if exists(self.VD_opt.scaler):
                pkg['VD_scaler'] = self.VD_opt.scaler.state_dict()

        # 如果存在 EMA 生成器，则保存其状态字典
        if self.has_ema_generator:
            pkg['G_ema'] = self.G_ema.state_dict()

        # 使用 torch 保存模型参数字典到指定路径
        torch.save(pkg, str(path))
    # 从指定路径加载模型参数
    def load(self, path, strict = False):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()

        # 加载模型参数
        pkg = torch.load(str(path))

        # 检查加载的模型参数版本是否与当前版本一致
        if 'version' in pkg and pkg['version'] != __version__:
            print(f"trying to load from version {pkg['version']}")

        # 加载生成器和判别器的状态字典
        self.unwrapped_G.load_state_dict(pkg['G'], strict = strict)
        self.unwrapped_D.load_state_dict(pkg['D'], strict = strict)

        # 如果存在 VD 模型，则加载其状态字典
        if exists(self.VD):
            self.unwrapped_VD.load_state_dict(pkg['VD'], strict = strict)

        # 如果有 EMA 生成器，则加载其状态字典
        if self.has_ema_generator:
            self.G_ema.load_state_dict(pkg['G_ema'])

        # 如果模型参数中包含步数信息，则更新当前步数
        if 'steps' in pkg:
            self.steps.copy_(torch.tensor([pkg['steps']]))

        # 如果模型参数中包含优化器状态字典，则加载优化器状态
        if 'G_opt'not in pkg or 'D_opt' not in pkg:
            return

        try:
            # 加载生成器和判别器的优化器状态字典
            self.G_opt.load_state_dict(pkg['G_opt'])
            self.D_opt.load_state_dict(pkg['D_opt'])

            # 如果存在 VD 模型，则加载其优化器状态字典
            if exists(self.VD):
                self.VD_opt.load_state_dict(pkg['VD_opt'])

            # 如果模型参数中包含生成器的缩放器状态字典，则加载
            if 'G_scaler' in pkg and exists(self.G_opt.scaler):
                self.G_opt.scaler.load_state_dict(pkg['G_scaler'])

            # 如果模型参数中包含判别器的缩放器状态字典，则加载
            if 'D_scaler' in pkg and exists(self.D_opt.scaler):
                self.D_opt.scaler.load_state_dict(pkg['D_scaler'])

            # 如果模型参数中包含 VD 的缩放器状态字典，则加载
            if 'VD_scaler' in pkg and exists(self.VD_opt.scaler):
                self.VD_opt.scaler.load_state_dict(pkg['VD_scaler'])

        except Exception as e:
            # 加载优化器状态字典出错时打印错误信息
            self.print(f'unable to load optimizers {e.msg}- optimizer states will be reset')
            pass

    # 加速相关

    # 获取设备信息
    @property
    def device(self):
        return self.accelerator.device

    # 获取未包装的生成器模型
    @property
    def unwrapped_G(self):
        return self.accelerator.unwrap_model(self.G)

    # 获取未包装的判别器模型
    @property
    def unwrapped_D(self):
        return self.accelerator.unwrap_model(self.D)

    # 获取未包装的 VD 模型
    @property
    def unwrapped_VD(self):
        return self.accelerator.unwrap_model(self.VD)

    # 是否需要视觉辅助判别器
    @property
    def need_vision_aided_discriminator(self):
        return exists(self.VD) and self.vision_aided_divergence_loss_weight > 0.

    # 是否需要对比损失
    @property
    def need_contrastive_loss(self):
        return self.generator_contrastive_loss_weight > 0. and not self.unconditional

    # 打印信息
    def print(self, msg):
        self.accelerator.print(msg)

    # 是否分布式
    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    # 是否为主进程
    @property
    def is_main(self):
        return self.accelerator.is_main_process

    # 是否为本地主进程
    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    # 调整图像大小
    def resize_image_to(self, images, resolution):
        return F.interpolate(images, resolution, mode = self.resize_image_mode)

    # 设置数据加载器
    @beartype
    def set_dataloader(self, dl: DataLoader):
        assert not exists(self.train_dl), 'training dataloader has already been set'

        self.train_dl = dl
        self.train_dl_batch_size = dl.batch_size

        self.train_dl = self.accelerator.prepare(self.train_dl)

    # 生成函数

    @torch.inference_mode()
    def generate(self, *args, **kwargs):
        model = self.G_ema if self.has_ema_generator else self.G
        model.eval()
        return model(*args, **kwargs)

    # 创建 EMA 生成器

    def create_ema_generator(
        self,
        update_every = 10,
        update_after_step = 100,
        decay = 0.995
    ):
        if not self.is_main:
            return

        assert not self.has_ema_generator, 'EMA generator has already been created'

        self.G_ema = EMA(self.unwrapped_G, update_every = update_every, update_after_step = update_after_step, beta = decay)
        self.has_ema_generator = True
    # 生成传递给生成器的参数
    def generate_kwargs(self, dl_iter, batch_size):
        # 根据训练是否为上采样器或非条件性来确定传递给生成器的内容

        # 可能的文本参数字典
        maybe_text_kwargs = dict()
        if self.train_upsampler or not self.unconditional:
            assert exists(dl_iter)

            if self.unconditional:
                real_images = next(dl_iter)
            else:
                result = next(dl_iter)
                assert isinstance(result, tuple), 'dataset should return a tuple of two items for text conditioned training, (images: Tensor, texts: List[str])'
                real_images, texts = result

                maybe_text_kwargs['texts'] = texts[:batch_size]

            real_images = real_images.to(self.device)

        # 如果训练上采样生成器，则需要对真实图像进行下采样
        if self.train_upsampler:
            size = self.unwrapped_G.input_image_size
            lowres_real_images = F.interpolate(real_images, (size, size))

            G_kwargs = dict(lowres_image = lowres_real_images)
        else:
            assert exists(batch_size)

            G_kwargs = dict(batch_size = batch_size)

        # 创建噪声
        noise = torch.randn(batch_size, self.unwrapped_G.style_network.dim, device = self.device)

        G_kwargs.update(noise = noise)

        return G_kwargs, maybe_text_kwargs
    
    # 训练鉴别器的步骤
    @beartype
    def train_discriminator_step(
        self,
        dl_iter: Iterable,
        grad_accum_every = 1,
        apply_gradient_penalty = False,
        calc_multiscale_loss = True
    # 训练生成器的步骤
    def train_generator_step(
        self,
        batch_size = None,
        dl_iter: Optional[Iterable] = None,
        grad_accum_every = 1,
        calc_multiscale_loss = True
        ):
        # 初始化各种损失值
        total_divergence = 0.
        total_multiscale_divergence = 0. if calc_multiscale_loss else None
        total_vd_divergence = 0.
        contrastive_loss = 0.

        # 设置生成器和判别器为训练模式
        self.G.train()
        self.D.train()

        # 清空生成器和判别器的梯度
        self.D_opt.zero_grad()
        self.G_opt.zero_grad()

        # 初始化存储所有图像和文本的列表
        all_images = []
        all_texts = []

        for _ in range(grad_accum_every):

            # 生成器部分

            # 生成生成器所需的参数和文本参数
            G_kwargs, maybe_text_kwargs = self.generate_kwargs(dl_iter, batch_size)

            # 自动混合精度加速
            with self.accelerator.autocast():
                # 生成图像和 RGB 值
                images, rgbs = self.G(
                    **G_kwargs,
                    **maybe_text_kwargs,
                    return_all_rgbs = True
                )

                # 使用不同的数据增强方法
                if exists(self.diff_augment):
                    images, rgbs = self.diff_augment(images, rgbs)

                # 如果需要对比损失，累积所有图像和文本
                if self.need_contrastive_loss:
                    all_images.append(images)
                    all_texts.extend(maybe_text_kwargs['texts'])

                # 判别器部分

                # 获取判别器的输出
                logits, multiscale_logits, _ = self.D(
                    images,
                    rgbs,
                    **maybe_text_kwargs,
                    return_multiscale_outputs = calc_multiscale_loss,
                    calc_aux_loss = False
                )

                # 生成器的 Hinge 损失和判别器的多尺度输出
                divergence = generator_hinge_loss(logits)

                total_divergence += (divergence.item() / grad_accum_every)

                total_loss = divergence

                # 如果多尺度分歧损失权重大于 0 并且有多尺度输出
                if self.multiscale_divergence_loss_weight > 0. and len(multiscale_logits) > 0:
                    multiscale_divergence = 0.

                    for multiscale_logit in multiscale_logits:
                        multiscale_divergence = multiscale_divergence + generator_hinge_loss(multiscale_logit)

                    total_multiscale_divergence += (multiscale_divergence.item() / grad_accum_every)

                    total_loss = total_loss + multiscale_divergence * self.multiscale_divergence_loss_weight

                # 视觉辅助生成器的 Hinge 损失
                if self.need_vision_aided_discriminator:
                    vd_loss = 0.

                    logits = self.VD(images, **maybe_text_kwargs)

                    for logit in logits:
                        vd_loss = vd_loss + generator_hinge_loss(logit)

                    total_vd_divergence += (vd_loss.item() / grad_accum_every)

                    total_loss = total_loss + vd_loss * self.vision_aided_divergence_loss_weight

            # 反向传播
            self.accelerator.backward(total_loss / grad_accum_every, retain_graph = self.need_contrastive_loss)

        # 如果需要生成器对比损失
        # 收集所有图像和文本并计算损失
        if self.need_contrastive_loss:
            all_images = torch.cat(all_images, dim = 0)

            contrastive_loss = aux_clip_loss(
                clip = self.G.text_encoder.clip,
                texts = all_texts,
                images = all_images
            )

            self.accelerator.backward(contrastive_loss * self.generator_contrastive_loss_weight)

        # 生成器优化器步骤
        self.G_opt.step()

        # 更新指数移动平均生成器
        self.accelerator.wait_for_everyone()

        if self.is_main and self.has_ema_generator:
            self.G_ema.update()

        # 返回训练生成器的损失
        return TrainGenLosses(
            total_divergence,
            total_multiscale_divergence,
            total_vd_divergence,
            contrastive_loss
        )
    # 定义一个方法用于生成样本，接受模型、数据迭代器和批量大小作为参数
    def sample(self, model, dl_iter, batch_size):
        # 生成生成器参数和可能的文本参数
        G_kwargs, maybe_text_kwargs = self.generate_kwargs(dl_iter, batch_size)

        # 使用加速器自动混合精度
        with self.accelerator.autocast():
            # 调用模型生成输出
            generator_output = model(**G_kwargs, **maybe_text_kwargs)

        # 如果不需要训练上采样器，则直接返回生成器输出
        if not self.train_upsampler:
            return generator_output

        # 获取生成器输出的大小
        output_size = generator_output.shape[-1]
        # 获取低分辨率图像
        lowres_image = G_kwargs['lowres_image']
        # 将低分辨率图像插值到与生成器输出相同的大小
        lowres_image = F.interpolate(lowres_image, (output_size, output_size))

        # 返回拼接后的图像
        return torch.cat([lowres_image, generator_output])

    # 进入推断模式的装饰器
    @torch.inference_mode()
    # 定义一个保存样本的方法，接受批量大小和数据迭代器作为参数
    def save_sample(
        self,
        batch_size,
        dl_iter = None
    ):
        # 计算当前里程碑
        milestone = self.steps.item() // self.save_and_sample_every
        # 如果训练上采样器，则设置 nrow_mult 为 2，否则为 1
        nrow_mult = 2 if self.train_upsampler else 1
        # 将样本数量分组成批次
        batches = num_to_groups(self.num_samples, batch_size)

        # 如果训练上采样器，则使用默认的上采样器数据迭代器
        if self.train_upsampler:
            dl_iter = default(self.sample_upsampler_dl_iter, dl_iter)

        # 断言数据迭代器存在
        assert exists(dl_iter)

        # 定义保存模型和输出文件名的列表
        sample_models_and_output_file_name = [(self.unwrapped_G, f'sample-{milestone}.png')]

        # 如果有 EMA 生成器，则添加到列表中
        if self.has_ema_generator:
            sample_models_and_output_file_name.append((self.G_ema, f'ema-sample-{milestone}.png'))

        # 遍历模型和文件名列表
        for model, filename in sample_models_and_output_file_name:
            # 将模型设置为评估模式
            model.eval()

            # 获取所有图像列表
            all_images_list = list(map(lambda n: self.sample(model, dl_iter, n), batches))
            # 拼接所有图像
            all_images = torch.cat(all_images_list, dim = 0)

            # 将图像像素值限制在 0 到 1 之间
            all_images.clamp_(0., 1.)

            # 保存图像
            utils.save_image(
                all_images,
                str(self.results_folder / filename),
                nrow = int(sqrt(self.num_samples)) * nrow_mult
            )

        # 可能的操作：包括一些指标以保存改进的内容，包括一些采样器字典文本条目
        # 保存模型
        self.save(str(self.model_folder / f'model-{milestone}.ckpt'))

    # 使用 beartype 装饰器定义前向传播方法，接受步数和梯度累积频率作为参数
    @beartype
    def forward(
        self,
        *,
        steps,
        grad_accum_every = 1
        ):
        # 断言训练数据加载器已设置，否则提示需要通过运行.set_dataloader(dl: Dataloader)来设置数据加载器
        assert exists(self.train_dl), 'you need to set the dataloader by running .set_dataloader(dl: Dataloader)'

        # 获取训练数据加载器的批量大小
        batch_size = self.train_dl_batch_size
        # 创建数据加载器的迭代器
        dl_iter = cycle(self.train_dl)

        # 初始化上一次的梯度惩罚损失、多尺度判别器损失和多尺度生成器损失
        last_gp_loss = 0.
        last_multiscale_d_loss = 0.
        last_multiscale_g_loss = 0.

        # 循环执行训练步骤
        for _ in tqdm(range(steps), initial = self.steps.item()):
            # 获取当前步骤数
            steps = self.steps.item()
            # 判断是否为第一步
            is_first_step = steps == 1

            # 判断是否需要应用梯度惩罚
            apply_gradient_penalty = self.apply_gradient_penalty_every > 0 and divisible_by(steps, self.apply_gradient_penalty_every)
            # 判断是否需要计算多尺度损失
            calc_multiscale_loss =  self.calc_multiscale_loss_every > 0 and divisible_by(steps, self.calc_multiscale_loss_every)

            # 调用训练判别器步骤函数，获取各种损失值
            (
                d_loss,
                multiscale_d_loss,
                vision_aided_d_loss,
                matching_aware_loss,
                gp_loss,
                recon_loss
            ) = self.train_discriminator_step(
                dl_iter = dl_iter,
                grad_accum_every = grad_accum_every,
                apply_gradient_penalty = apply_gradient_penalty,
                calc_multiscale_loss = calc_multiscale_loss
            )

            # 等待所有进程完成
            self.accelerator.wait_for_everyone()

            # 调用训练生成器步骤函数，获取各种损失值
            (
                g_loss,
                multiscale_g_loss,
                vision_aided_g_loss,
                contrastive_loss
            ) = self.train_generator_step(
                dl_iter = dl_iter,
                batch_size = batch_size,
                grad_accum_every = grad_accum_every,
                calc_multiscale_loss = calc_multiscale_loss
            )

            # 如果梯度惩罚损失存在，则更新上一次的梯度惩罚损失
            if exists(gp_loss):
                last_gp_loss = gp_loss

            # 如果多尺度判别器损失存在，则更新上一次的多尺度判别器损失
            if exists(multiscale_d_loss):
                last_multiscale_d_loss = multiscale_d_loss

            # 如果多尺度生成器损失存在，则更新上一次的多尺度生成器损失
            if exists(multiscale_g_loss):
                last_multiscale_g_loss = multiscale_g_loss

            # 如果是第一步或者步骤数能被log_steps_every整除，则输出损失信息
            if is_first_step or divisible_by(steps, self.log_steps_every):

                # 构建损失信息元组
                losses = (
                    ('G', g_loss),
                    ('MSG', last_multiscale_g_loss),
                    ('VG', vision_aided_g_loss),
                    ('D', d_loss),
                    ('MSD', last_multiscale_d_loss),
                    ('VD', vision_aided_d_loss),
                    ('GP', last_gp_loss),
                    ('SSL', recon_loss),
                    ('CL', contrastive_loss),
                    ('MAL', matching_aware_loss)
                )

                # 将损失信息转换为字符串格式
                losses_str = ' | '.join([f'{loss_name}: {loss:.2f}' for loss_name, loss in losses])

                # 打印损失信息
                self.print(losses_str)

            # 等待所有进程完成
            self.accelerator.wait_for_everyone()

            # 如果是主进程且是第一步或者步骤数能被save_and_sample_every整除或者步骤数小于early_save_thres_steps且能被early_save_and_sample_every整除，则保存样本
            if self.is_main and (is_first_step or divisible_by(steps, self.save_and_sample_every) or (steps <= self.early_save_thres_steps and divisible_by(steps, self.early_save_and_sample_every))):
                self.save_sample(batch_size, dl_iter)
            
            # 更新步骤数
            self.steps += 1

        # 打印完成训练步骤数
        self.print(f'complete {steps} training steps')
```