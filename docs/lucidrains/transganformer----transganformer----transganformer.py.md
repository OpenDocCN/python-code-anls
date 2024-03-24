# `.\lucidrains\transganformer\transganformer\transganformer.py`

```
# 导入所需的库
import os
import json
import multiprocessing
from random import random
import math
from math import log2, floor, sqrt, log, pi
from functools import partial
from contextlib import contextmanager, ExitStack
from pathlib import Path
from shutil import rmtree

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
import torchvision
from torchvision import transforms
from kornia import filter2D

from transganformer.diff_augment import DiffAugment
from transganformer.version import __version__

from tqdm import tqdm
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

# 断言CUDA是否可用
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# 常量定义
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 空上下文管理器
@contextmanager
def null_context():
    yield

# 合并多个上下文管理器
def combine_contexts(contexts):
    @contextmanager
    def multi_contexts():
        with ExitStack() as stack:
            yield [stack.enter_context(ctx()) for ctx in contexts]
    return multi_contexts

# 判断是否为2的幂
def is_power_of_two(val):
    return log2(val).is_integer()

# 设置模型参数是否需要梯度
def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

# 无限循环生成器
def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# 如果值为NaN，则抛出异常
def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

# 梯度累积上下文管理器
def gradient_accumulate_contexts(gradient_accumulate_every, is_ddp, ddps):
    if is_ddp:
        num_no_syncs = gradient_accumulate_every - 1
        head = [combine_contexts(map(lambda ddp: ddp.no_sync, ddps))] * num_no_syncs
        tail = [null_context]
        contexts =  head + tail
    else:
        contexts = [null_context] * gradient_accumulate_every

    for context in contexts:
        with context():
            yield

# 分块评估
def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

# 球面线性插值
def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# 安全除法
def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = '' if int(n >= 0) else '-'
        res = float(f'{prefix}inf')
    return res

# 辅助类

# NaN异常类
class NanException(Exception):
    pass

# 指数移动平均类
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new

# 随机应用类
class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

# 残差连接类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    # 定义一个前向传播函数，接受输入 x 和其他关键字参数
    def forward(self, x, **kwargs):
        # 调用 self.fn 函数进行前向传播，得到输出 out
        out = self.fn(x, **kwargs)

        # 如果输出是一个元组
        if isinstance(out, tuple):
            # 将元组拆分为 out 和 latent 两部分
            out, latent = out
            # 将输入 x 和 out 相加，得到 ret
            ret = (out + x, latent)
            # 返回 ret
            return ret

        # 如果输出不是元组，则将输入 x 和输出 out 相加，返回结果
        return x + out
class SumBranches(nn.Module):
    # 定义一个类，用于将多个分支的输出求和
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        # 对每个分支的输出进行映射并求和
        return sum(map(lambda fn: fn(x), self.branches))

# attention and transformer modules

class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # 计算输入张量 x 的标准差和均值，进行归一化处理
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn, dim_context = None):
        super().__init__()
        self.norm = ChanNorm(dim)
        self.norm_context = ChanNorm(dim_context) if exists(dim_context) else None
        self.fn = fn

    def forward(self, x, **kwargs):
        # 对输入张量 x 进行归一化处理
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs.pop('context')
            context = self.norm_context(context)
            kwargs.update(context = context)

        return self.fn(x, **kwargs)

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

def FeedForward(dim, mult = 4, kernel_size = 3, bn = False):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(dim, dim * mult * 2, 1),
        nn.GLU(dim = 1),
        nn.BatchNorm2d(dim * mult) if bn else nn.Identity(),
        DepthWiseConv2d(dim * mult, dim * mult * 2, kernel_size, padding = padding),
        nn.GLU(dim = 1),
        nn.Conv2d(dim * mult, dim, 1)
    )

# sinusoidal embedding

class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        dim //= 2
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        h = torch.linspace(-1., 1., x.shape[-2], device = x.device).type_as(self.inv_freq)
        w = torch.linspace(-1., 1., x.shape[-1], device = x.device).type_as(self.inv_freq)
        sinu_inp_h = torch.einsum('i , j -> i j', h, self.inv_freq)
        sinu_inp_w = torch.einsum('i , j -> i j', w, self.inv_freq)
        sinu_inp_h = repeat(sinu_inp_h, 'h c -> () c h w', w = x.shape[-1])
        sinu_inp_w = repeat(sinu_inp_w, 'w c -> () c h w', h = x.shape[-2])
        sinu_inp = torch.cat((sinu_inp_w, sinu_inp_h), dim = 1)
        emb = torch.cat((sinu_inp.sin(), sinu_inp.cos()), dim = 1)
        return emb

# classes

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        fmap_size = None,
        dim_out = None,
        kv_dim = None,
        heads = 8,
        dim_head = 64,
        q_kernel_size = 1,
        kv_kernel_size = 3,
        out_kernel_size = 1,
        q_stride = 1,
        include_self = False,
        downsample = False,
        downsample_kv = 1,
        bn = False,
        latent_dim = None
        ):
        # 调用父类的构造函数
        super().__init__()
        # 创建固定位置嵌入对象
        self.sinu_emb = FixedPositionalEmbedding(dim)

        # 计算内部维度
        inner_dim = dim_head *  heads
        # 设置键值维度，默认为 dim
        kv_dim = default(kv_dim, dim)
        # 设置输出维度，默认为 dim
        dim_out = default(dim_out, dim)

        # 设置头数和缩放因子
        self.heads = heads
        self.scale = dim_head ** -0.5

        # 计算填充值
        q_padding = q_kernel_size // 2
        kv_padding = kv_kernel_size // 2
        out_padding = out_kernel_size // 2

        # 设置查询卷积参数
        q_conv_params = (1, 1, 0)

        # 创建查询卷积层
        self.to_q = nn.Conv2d(dim, inner_dim, *q_conv_params, bias = False)

        # 根据下采样因子设置键值卷积参数
        if downsample_kv == 1:
            kv_conv_params = (3, 1, 1)
        elif downsample_kv == 2:
            kv_conv_params = (3, 2, 1)
        elif downsample_kv == 4:
            kv_conv_params = (7, 4, 3)
        else:
            raise ValueError(f'invalid downsample factor for key / values {downsample_kv}')

        # 创建键卷积层和值卷积层
        self.to_k = nn.Conv2d(kv_dim, inner_dim, *kv_conv_params, bias = False)
        self.to_v = nn.Conv2d(kv_dim, inner_dim, *kv_conv_params, bias = False)

        # 设置是否使用批归一化
        self.bn = bn
        if self.bn:
            self.q_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()
            self.k_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()
            self.v_bn = nn.BatchNorm2d(inner_dim) if bn else nn.Identity()

        # 检查是否存在潜在维度
        self.has_latents = exists(latent_dim)
        if self.has_latents:
            # 创建潜在维度的通道归一化层和潜在维度到查询、键、值的卷积层
            self.latent_norm = ChanNorm(latent_dim)
            self.latents_to_qkv = nn.Conv2d(latent_dim, inner_dim * 3, 1, bias = False)

            # 创建潜在维度到输出的卷积层序列
            self.latents_to_out = nn.Sequential(
                nn.Conv2d(inner_dim, latent_dim * 2, 1),
                nn.GLU(dim = 1),
                nn.BatchNorm2d(latent_dim) if bn else nn.Identity()
            )

        # 设置是否包含自身
        self.include_self = include_self
        if include_self:
            # 创建自身到自身的键卷积层和值卷积层
            self.to_self_k = nn.Conv2d(dim, inner_dim, *kv_conv_params, bias = False)
            self.to_self_v = nn.Conv2d(dim, inner_dim, *kv_conv_params, bias = False)

        # 创建混合头部后的参数
        self.mix_heads_post = nn.Parameter(torch.randn(heads, heads))

        # 根据是否下采样设置输出卷积参数
        out_conv_params = (3, 2, 1) if downsample else q_conv_params

        # 创建输出卷积层序列
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim_out * 2, *out_conv_params),
            nn.GLU(dim = 1),
            nn.BatchNorm2d(dim_out) if bn else nn.Identity()
        )

        # 设置特征图大小和旋转嵌入
        self.fmap_size = fmap_size
        self.pos_emb = RotaryEmbedding(dim_head, downsample_keys = downsample_kv)
    # 定义前向传播函数，接受输入 x，潜在变量 latents，默认上下文 context，是否包含自身 include_self
    def forward(self, x, latents = None, context = None, include_self = False):
        # 断言检查输入 x 的最后一个维度是否与指定的 fmap_size 相等
        assert not exists(self.fmap_size) or x.shape[-1] == self.fmap_size, 'fmap size must equal the given shape'

        # 获取输入 x 的形状信息
        b, n, _, y, h, device = *x.shape, self.heads, x.device

        # 检查是否存在上下文信息，如果不存在，则使用输入 x 作为上下文
        has_context = exists(context)
        context = default(context, x)

        # 初始化查询、键、值的输入
        q_inp = x
        k_inp = context
        v_inp = context

        # 如果不存在上下文信息，则添加正弦嵌入
        if not has_context:
            sinu_emb = self.sinu_emb(context)
            q_inp += sinu_emb
            k_inp += sinu_emb

        # 将查询、键、值通过对应的线性变换层
        q, k, v = (self.to_q(q_inp), self.to_k(k_inp), self.to_v(v_inp))

        # 如果启用了批归一化，则对查询、键、值进行批归一化
        if self.bn:
            q = self.q_bn(q)
            k = self.k_bn(k)
            v = self.v_bn(v)

        # 获取查询的输出高度和宽度
        out_h, out_w = q.shape[-2:]

        # 定义函数将查询、键、值按头数拆分
        split_head = lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = h)

        # 对查询、键、值按头数拆分
        q, k, v = map(split_head, (q, k, v))

        # 如果不存在上下文信息，则对查询、键添加位置嵌入
        if not has_context:
            q, k = self.pos_emb(q, k)

        # 如果包含自身信息，则将自身信息添加到键和值中
        if self.include_self:
            kx = self.to_self_k(x)
            vx = self.to_self_v(x)
            kx, vx = map(split_head, (kx, vx))

            k = torch.cat((kx, k), dim = -2)
            v = torch.cat((vx, v), dim = -2)

        # 如果存在潜在变量，则将潜在变量信息添加到查询、键、值中
        if self.has_latents:
            assert exists(latents), 'latents must be passed in'
            latents = self.latent_norm(latents)
            lq, lk, lv = self.latents_to_qkv(latents).chunk(3, dim = 1)
            lq, lk, lv = map(split_head, (lq, lk, lv))

            latent_shape = lq.shape
            num_latents = lq.shape[-2]

            q = torch.cat((lq, q), dim = -2)
            k = torch.cat((lk, k), dim = -2)
            v = torch.cat((lv, v), dim = -2)

        # 计算点积注意力得分
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 对注意力得分进行 softmax 操作
        attn = dots.softmax(dim = -1)
        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post)

        # 计算输出
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # 如果存在潜在变量，则将潜在变量信息分离出来
        if self.has_latents:
            lout, out = out[..., :num_latents, :], out[..., num_latents:, :]
            lout = rearrange(lout, 'b h (x y) d -> b (h d) x y', h = h, x = latents.shape[-2], y = latents.shape[-1])
            lout = self.latents_to_out(lout)

        # 重组输出形状
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h = h, x = out_h, y = out_w)
        out = self.to_out(out)

        # 如果存在潜在变量，则返回输出和潜在变量输出
        if self.has_latents:
            return out, lout

        # 否则只返回输出
        return out
# dataset

# 将图像转换为指定类型
def convert_image_to(img_type, image):
    # 如果图像模式不是指定类型，则进行转换
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 定义一个身份函数类
class identity(object):
    def __call__(self, tensor):
        return tensor

# 扩展灰度图像类
class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        # 获取图像通道数
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        # 如果通道数与目标通道数相同，则返回原图像
        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f'image with invalid number of channels given {channels}')

        # 如果不存在 alpha 通道且需要透明度，则创建全白的 alpha 通道
        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

# 调整图像大小至最小尺寸
def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

# 图像数据集类
class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        transparent = False,
        greyscale = False,
        aug_prob = 0.
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        if transparent:
            num_channels = 4
            pillow_mode = 'RGBA'
            expand_fn = expand_greyscale(transparent)
        elif greyscale:
            num_channels = 1
            pillow_mode = 'L'
            expand_fn = identity()
        else:
            num_channels = 3
            pillow_mode = 'RGB'
            expand_fn = expand_greyscale(transparent)

        convert_image_fn = partial(convert_image_to, pillow_mode)

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_fn)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# augmentations

# 随机水平翻转函数
def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))

# 增强包装类
class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False, **kwargs):
        context = torch.no_grad if detach else null_context

        with context():
            if random() < prob:
                images = random_hflip(images, prob=0.5)
                images = DiffAugment(images, types=types)

        return self.D(images, **kwargs)

# modifiable global variables

# 上采样函数
def upsample(scale_factor = 2):
    return nn.Upsample(scale_factor = scale_factor)

# activation

# Leaky ReLU 激活函数
def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

# rotary positional embedding helpers

# 每两个元素旋转函数
def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

# 获取正弦余弦值函数
def get_sin_cos(seq):
    n = seq.shape[0]
    x_sinu = repeat(seq, 'i d -> i j d', j = n)
    y_sinu = repeat(seq, 'j d -> i j d', i = n)

    sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
    # 将 x_sinu 和 y_sinu 的余弦值按照最后一个维度连接起来
    cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

    # 对 sin 和 cos 进行重排列，将最后两个维度合并到一起
    sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
    # 对 sin 和 cos 进行重复，扩展维度
    sin, cos = map(lambda t: repeat(t, 'n d -> () () n (d j)', j = 2), (sin, cos))
    # 返回重排列后的 sin 和 cos
    return sin, cos
# positional encoding

# 定义旋转嵌入类
class RotaryEmbedding(nn.Module):
    # 初始化函数
    def __init__(self, dim, downsample_keys = 1):
        super().__init__()
        self.dim = dim
        self.downsample_keys = downsample_keys

    # 前向传播函数
    def forward(self, q, k):
        device, dtype, n = q.device, q.dtype, int(sqrt(q.shape[-2]))

        # 生成等间距序列
        seq = torch.linspace(-1., 1., steps = n, device = device)
        seq = seq.unsqueeze(-1)

        # 生成不同尺度的旋转角度
        scales = torch.logspace(0., log(10 / 2) / log(2), self.dim // 4, base = 2, device = device, dtype = dtype)
        scales = scales[(*((None,) * (len(seq.shape) - 1)), Ellipsis]

        seq = seq * scales * pi

        x = seq
        y = seq

        # 对 y 进行降采样
        y = reduce(y, '(j n) c -> j c', 'mean', n = self.downsample_keys)

        # 获取正弦和余弦值
        q_sin, q_cos = get_sin_cos(x)
        k_sin, k_cos = get_sin_cos(y)
        q = (q * q_cos) + (rotate_every_two(q) * q_sin)
        k = (k * k_cos) + (rotate_every_two(k) * k_sin)
        return q, k

# mapping network

# 定义等权重线性变换类
class EqualLinear(nn.Module):
    # 初始化函数
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    # 前向传播函数
    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

# 定义映射网络类
class MappingNetwork(nn.Module):
    # 初始化函数
    def __init__(self, dim, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(dim, dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(
            *layers,
            nn.Linear(dim, dim * 4)
        )

    # 前向传播函数
    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.net(x)
        return rearrange(x, 'b (c h w) -> b c h w', h = 2, w = 2)

# generative adversarial network

# 定义生成器类
class Generator(nn.Module):
    # 初始化函数
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        fmap_max = 512,
        init_channel = 3,
        mapping_network_depth = 4
    ):
        super().__init__()
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        num_layers = int(log2(image_size)) - 1
        
        self.mapping = MappingNetwork(latent_dim, mapping_network_depth)
        self.initial_block = nn.Parameter(torch.randn((latent_dim, 4, 4)))

        self.layers = nn.ModuleList([])

        fmap_size = 4
        chan = latent_dim
        min_chan = 8

        for ind in range(num_layers):
            is_last = ind == (num_layers - 1)

            downsample_factor = int(2 ** max(log2(fmap_size) - log2(32), 0))
            attn_class = partial(Attention, bn = True, fmap_size = fmap_size, downsample_kv = downsample_factor)

            if not is_last:
                chan_out = max(min_chan, chan // 4)

                upsample = nn.Sequential(
                    attn_class(dim = chan, dim_head = chan, heads = 1, dim_out = chan_out * 4),
                    nn.PixelShuffle(2)
                )

            else:
                upsample = nn.Identity()

            self.layers.append(nn.ModuleList([
                Residual(PreNorm(chan, attn_class(dim = chan, latent_dim = latent_dim))),
                Residual(FeedForward(chan, bn = True, kernel_size = (3 if image_size > 4 else 1))),
                upsample,
            ]))

            chan = chan_out
            fmap_size *= 2

        self.final_attn = Residual(PreNorm(chan, attn_class(chan, latent_dim = latent_dim)))

        self.to_img = nn.Sequential(
            Residual(FeedForward(chan_out, bn = True)),
            nn.Conv2d(chan, init_channel, 1)
        )
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 获取输入 x 的 batch 大小
        b = x.shape[0]

        # 将输入 x 映射到潜在空间
        latents = self.mapping(x)

        # 重复初始块的特征图，使其与 batch 大小相匹配
        fmap = repeat(self.initial_block, 'c h w -> b c h w', b = b)

        # 遍历每个层中的注意力机制、特征提取和上采样操作
        for attn, ff, upsample in self.layers:
            # 使用注意力机制处理特征图和潜在空间
            fmap, latents_out = attn(fmap, latents = latents)
            # 更新潜在空间
            latents = latents + latents_out

            # 使用特征提取函数处理特征图
            fmap = ff(fmap)
            # 使用上采样函数对特征图进行上采样

            fmap = upsample(fmap)

        # 最终使用最终的注意力机制处理特征图和潜在空间
        fmap, _ = self.final_attn(fmap, latents = latents)
        # 将处理后的特征图转换为图像
        return self.to_img(fmap)
# 定义一个简单的解码器类，继承自 nn.Module
class SimpleDecoder(nn.Module):
    # 初始化函数，设置输入通道数、输出通道数、上采样次数等参数
    def __init__(
        self,
        *,
        chan_in,
        chan_out = 3,
        num_upsamples = 4,
    ):
        super().__init__()

        # 初始化空的层列表
        layers = nn.ModuleList([])
        # 设置最终输出通道数
        final_chan = chan_out
        # 设置初始通道数
        chans = chan_in

        # 循环创建上采样层
        for ind in range(num_upsamples):
            # 判断是否是最后一层
            last_layer = ind == (num_upsamples - 1)
            # 根据是否是最后一层确定输出通道数
            chan_out = chans if not last_layer else final_chan * 2
            # 创建包含上采样、卷积和 GLU 激活函数的层
            layer = nn.Sequential(
                upsample(),
                nn.Conv2d(chans, chan_out, 3, padding = 1),
                nn.GLU(dim = 1)
            )
            # 将层添加到层列表中
            layers.append(layer)
            # 更新通道数
            chans //= 2

        # 将所有层组合成一个网络
        self.net = nn.Sequential(*layers)

    # 前向传播函数
    def forward(self, x):
        return self.net(x)

# 定义一个鉴别器类，继承自 nn.Module
class Discriminator(nn.Module):
    # 初始化函数，设置图像大小、最大特征图数、初始通道数等参数
    def __init__(
        self,
        *,
        image_size,
        fmap_max = 256,
        init_channel = 3,
    ):
        super().__init__()
        # 断言图像大小为 2 的幂次方
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        # 计算层数
        num_layers = int(log2(image_size)) - 2
        # 设置特征图维度
        fmap_dim = 64

        # 创建卷积嵌入层
        self.conv_embed = nn.Sequential(
            nn.Conv2d(init_channel, 32, kernel_size = 4, stride = 2, padding = 1),
            nn.Conv2d(32, fmap_dim, kernel_size = 3, padding = 1)
        )

        # 更新图像大小
        image_size //= 2
        # 创建横向和纵向位置嵌入参数
        self.ax_pos_emb_h = nn.Parameter(torch.randn(image_size, fmap_dim))
        self.ax_pos_emb_w = nn.Parameter(torch.randn(image_size, fmap_dim))

        # 初始化空的图层列表和特征图维度列表
        self.image_sizes = []
        self.layers = nn.ModuleList([])
        fmap_dims = []

        # 循环创建图层
        for ind in range(num_layers):
            # 更新图像大小
            image_size //= 2
            self.image_sizes.append(image_size)

            # 计算输出特征图维度
            fmap_dim_out = min(fmap_dim * 2, fmap_max)

            # 创建下采样分支
            downsample = SumBranches([
                nn.Conv2d(fmap_dim, fmap_dim_out, 3, 2, 1),
                nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(fmap_dim, fmap_dim_out, 3, padding = 1),
                    leaky_relu()
                )
            ])

            # 计算下采样因子
            downsample_factor = 2 ** max(log2(image_size) - log2(32), 0)
            # 创建注意力类
            attn_class = partial(Attention, fmap_size = image_size, downsample_kv = downsample_factor)

            # 将下采样、残差块和前馈网络块组合成一个图层
            self.layers.append(nn.ModuleList([
                downsample,
                Residual(PreNorm(fmap_dim_out, attn_class(dim = fmap_dim_out))),
                Residual(PreNorm(fmap_dim_out, FeedForward(dim = fmap_dim_out, kernel_size = (3 if image_size > 4 else 1)))
            ]))

            # 更新特征图维度和特征图维度列表
            fmap_dim = fmap_dim_out
            fmap_dims.append(fmap_dim)

        # 创建辅助解码器
        self.aux_decoder = SimpleDecoder(chan_in = fmap_dims[-2], chan_out = init_channel, num_upsamples = num_layers)

        # 创建输出层
        self.to_logits = nn.Sequential(
            Residual(PreNorm(fmap_dim, Attention(dim = fmap_dim, fmap_size = 2))),
            Residual(PreNorm(fmap_dim, FeedForward(dim = fmap_dim, kernel_size = (3 if image_size > 64 else 1)))),
            nn.Conv2d(fmap_dim, 1, 2),
            Rearrange('b () () () -> b')
        )

    # 前向传播函数
    def forward(self, x, calc_aux_loss = False):
        x_ = x
        x = self.conv_embed(x)

        ax_pos_emb = rearrange(self.ax_pos_emb_h, 'h c -> () c h ()') + rearrange(self.ax_pos_emb_w, 'w c -> () c () w')
        x += ax_pos_emb

        fmaps = []

        for (downsample, attn, ff), image_size in zip(self.layers, self.image_sizes):
            x = downsample(x)
            x = attn(x)
            x = ff(x)

            fmaps.append(x)

        x = self.to_logits(x)

        if not calc_aux_loss:
            return x, None

        recon = self.aux_decoder(fmaps[-2])
        recon_loss = F.mse_loss(x_, recon)
        return x, recon_loss

# 定义一个 Transganformer 类，继承自 nn.Module
class Transganformer(nn.Module):
    # 初始化函数，设置潜在维度、图像大小、最大特征图数等参数
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        fmap_max = 512,
        transparent = False,
        greyscale = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 初始化潜在空间维度和图像大小
        self.latent_dim = latent_dim
        self.image_size = image_size

        # 根据是否透明或灰度图像确定初始通道数
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        # 创建生成器参数字典
        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            fmap_max = fmap_max,
            init_channel = init_channel
        )

        # 初始化生成器和判别器
        self.G = Generator(**G_kwargs)
        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            init_channel = init_channel
        )

        # 初始化指数移动平均更新器和生成器EMA
        self.ema_updater = EMA(0.995)
        self.GE = Generator(**G_kwargs)
        set_requires_grad(self.GE, False)

        # 初始化生成器和判别器的优化器
        self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))

        # 初始化权重
        self.apply(self._init_weights)
        self.reset_parameter_averaging()

        # 将模型移至GPU
        self.cuda(rank)
        # 初始化带数据增强的判别器
        self.D_aug = AugWrapper(self.D, image_size)

    # 初始化权重函数
    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    # 更新EMA函数
    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

            for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
                new_buffer_value = self.ema_updater.update_average(ma_buffer, current_buffer)
                ma_buffer.copy_(new_buffer_value)

        update_moving_average(self.GE, self.G)

    # 重置参数平均函数
    def reset_parameter_averaging(self):
        self.GE.load_state_dict(self.G.state_dict())

    # 前向传播函数
    def forward(self, x):
        raise NotImplemented
# 定义 Trainer 类，用于训练模型
class Trainer():
    # 初始化函数，设置各种参数
    def __init__(
        self,
        name = 'default',
        results_dir = 'results',
        models_dir = 'models',
        base_dir = './',
        num_workers = None,
        latent_dim = 256,
        image_size = 128,
        num_image_tiles = 8,
        fmap_max = 512,
        transparent = False,
        greyscale = False,
        batch_size = 4,
        gp_weight = 10,
        gradient_accumulate_every = 1,
        lr = 2e-4,
        lr_mlp = 1.,
        ttur_mult = 1.,
        save_every = 1000,
        evaluate_every = 1000,
        aug_prob = None,
        aug_types = ['translation', 'cutout'],
        dataset_aug_prob = 0.,
        calculate_fid_every = None,
        calculate_fid_num_images = 12800,
        clear_fid_cache = False,
        is_ddp = False,
        rank = 0,
        world_size = 1,
        log = False,
        amp = False,
        *args,
        **kwargs
    ):
        # 存储传入的参数
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name

        # 设置路径相关参数
        base_dir = Path(base_dir)
        self.base_dir = base_dir
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name

        self.config_path = self.models_dir / name / '.config.json'

        # 检查图片大小是否为2的幂次方
        assert is_power_of_two(image_size), 'image size must be a power of 2 (32, 64, 128, 256, 512, 1024)'

        # 设置图片相关参数
        self.image_size = image_size
        self.num_image_tiles = num_image_tiles

        # 设置潜在空间维度、特征图最大值、透明度、灰度等参数
        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.transparent = transparent
        self.greyscale = greyscale

        # 检查透明度和灰度是否只设置了一个
        assert (int(self.transparent) + int(self.greyscale)) < 2, 'you can only set either transparency or greyscale'

        # 设置数据增强相关参数
        self.aug_prob = aug_prob
        self.aug_types = aug_types

        # 设置学习率、工作进程数、TTUR倍数、批量大小、梯度积累等参数
        self.lr = lr
        self.num_workers = num_workers
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        # 设置梯度惩罚权重
        self.gp_weight = gp_weight

        # 设置评估和保存模型的频率
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        # 初始化损失值
        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        # 初始化文件夹
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        # 设置计算 FID 的频率和数量
        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        # 设置是否使用分布式训练
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        # 设置混合精度训练
        self.amp = amp
        self.G_scaler = GradScaler(enabled = self.amp)
        self.D_scaler = GradScaler(enabled = self.amp)

    # 返回图片扩展名
    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    # 返回检查点编号
    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
        
    # 初始化 GAN 模型
    def init_GAN(self):
        args, kwargs = self.GAN_params

        # 实例化 GAN 模型
        self.GAN = Transganformer(
            lr = self.lr,
            latent_dim = self.latent_dim,
            image_size = self.image_size,
            ttur_mult = self.ttur_mult,
            fmap_max = self.fmap_max,
            transparent = self.transparent,
            greyscale = self.greyscale,
            rank = self.rank,
            *args,
            **kwargs
        )

        # 如果使用分布式训练，设置相关参数
        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    # 写入配置文件
    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))
    # 加载配置信息，如果配置文件不存在则使用默认配置，否则读取配置文件内容
    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        # 设置图像大小和透明度
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        # 设置是否为灰度图像，并移除配置中的灰度信息
        self.greyscale = config.pop('greyscale', False)
        # 移除配置中的 fmap_max 信息
        self.fmap_max = config.pop('fmap_max', 512)
        # 删除 GAN 属性
        del self.GAN
        # 初始化 GAN
        self.init_GAN()

    # 返回配置信息
    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
            'greyscale': self.greyscale
        }

    # 设置数据源文件夹
    def set_data_src(self, folder):
        # 计算默认的工作线程数
        num_workers = default(self.num_workers, math.ceil(NUM_CORES / self.world_size))
        # 创建图像数据集
        self.dataset = ImageDataset(folder, self.image_size, transparent=self.transparent, greyscale=self.greyscale, aug_prob=self.dataset_aug_prob)
        # 创建分布式采样器
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        # 创建数据加载器
        dataloader = DataLoader(self.dataset, num_workers=num_workers, batch_size=math.ceil(self.batch_size / self.world_size), sampler=sampler, shuffle=not self.is_ddp, drop_last=True, pin_memory=True)
        # 创建数据加载器的循环迭代器
        self.loader = cycle(dataloader)

        # 如果数据集较小，自动设置数据增强概率
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    # 评估生成器的效果
    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=4):
        self.GAN.eval()

        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # 生成潜在向量
        latents = torch.randn((num_rows ** 2, latent_dim)).cuda(self.rank)

        # 生成普通图像
        generated_images = self.generate_(self.GAN.G, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # 生成移动平均图像
        generated_images = self.generate_(self.GAN.GE, latents)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

    # 生成图像
    @torch.no_grad()
    def generate(self, num=0, num_image_tiles=4, checkpoint=None, types=['default', 'ema']):
        self.GAN.eval()

        latent_dim = self.GAN.latent_dim
        dir_name = self.name + str('-generated-') + str(checkpoint)
        dir_full = Path().absolute() / self.results_dir / dir_name
        ext = self.image_extension

        # 如果目录不存在，则创建目录
        if not dir_full.exists():
            os.mkdir(dir_full)

        # 生成普通图像
        if 'default' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated default images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        # 生成移动平均图像
        if 'ema' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated EMA images'):
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-ema.{ext}')
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        return dir_full

    @torch.no_grad()
    # 显示训练进度的方法，生成进度图像
    def show_progress(self, num_images=4, types=['default', 'ema']):
        # 获取所有检查点
        checkpoints = self.get_checkpoints()
        # 确保存在检查点以创建训练进度视频
        assert exists(checkpoints), 'cannot find any checkpoints to create a training progress video for'

        # 创建目录名
        dir_name = self.name + str('-progress')
        # 获取完整目录路径
        dir_full = Path().absolute() / self.results_dir / dir_name
        # 获取图像扩展名
        ext = self.image_extension
        # 初始化潜在向量
        latents = None

        # 计算检查点数的位数
        zfill_length = math.ceil(math.log10(len(checkpoints)))

        # 如果目录不存在，则创建目录
        if not dir_full.exists():
            os.mkdir(dir_full)

        # 遍历检查点，生成进度图像
        for checkpoint in tqdm(checkpoints, desc='Generating progress images'):
            # 加载检查点
            self.load(checkpoint, print_version=False)
            self.GAN.eval()

            # 初始化潜在向量
            if checkpoint == 0:
                latents = torch.randn((num_images, self.GAN.latent_dim)).cuda(self.rank)

            # 生成正常图像
            if 'default' in types:
                generated_image = self.generate_(self.GAN.G, latents)
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}.{ext}')
                torchvision.utils.save_image(generated_image, path, nrow=num_images)

            # 生成移动平均图像
            if 'ema' in types:
                generated_image = self.generate_(self.GAN.GE, latents)
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}-ema.{ext}')
                torchvision.utils.save_image(generated_image, path, nrow=num_images)

    # 计算 FID 分数的方法
    @torch.no_grad()
    def calculate_fid(self, num_batches):
        # 导入 FID 分数计算模块
        from pytorch_fid import fid_score
        # 清空 GPU 缓存
        torch.cuda.empty_cache()

        # 真实图像路径和生成图像路径
        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # 删除用于 FID 计算的现有文件并重新创建目录
        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            # 保存真实图像
            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    ind = k + batch_num * self.batch_size
                    torchvision.utils.save_image(image, real_path / f'{ind}.png')

        # 删除生成图像目录并重新创建
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        # 设置生成器为评估模式
        self.GAN.eval()
        ext = self.image_extension

        # 获取潜在向量维度和图像尺寸
        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # 生成假图像
        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # 生成潜在向量
            latents = torch.randn(self.batch_size, latent_dim).cuda(self.rank)

            # 生成移动平均图像
            generated_images = self.generate_(self.GAN.GE, latents)

            for j, image in enumerate(generated_images.unbind(0)):
                ind = j + batch_num * self.batch_size
                torchvision.utils.save_image(image, str(fake_path / f'{str(ind)}-ema.{ext}'))

        # 返回 FID 分数
        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, latents.device, 2048)

    # 生成图像的方法
    @torch.no_grad()
    def generate_(self, G, style, num_image_tiles = 8):
        # 分块评估生成图像
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    # 生成插值图像序列
    def generate_interpolation(self, num = 0, num_image_tiles = 8, num_steps = 100, save_frames = False):
        # 将 GAN 设置为评估模式
        self.GAN.eval()
        # 获取图像文件扩展名
        ext = self.image_extension
        # 设置图像行数
        num_rows = num_image_tiles

        # 获取潜在空间维度和图像尺寸
        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # 生成低和高潜在向量
        latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        # 生成插值比例
        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        # 对每个比例进行插值
        for ratio in tqdm(ratios):
            # 使用球面线性插值生成插值潜在向量
            interp_latents = slerp(ratio, latents_low, latents_high)
            # 生成图像
            generated_images = self.generate_(self.GAN.GE, interp_latents)
            # 将生成的图像排列成网格
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            # 将图像网格转换为 PIL 图像
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            # 如果需要透明背景
            if self.transparent:
                background = Image.new('RGBA', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            # 将当前帧添加到帧列表中
            frames.append(pil_image)

        # 保存插值图像序列为 GIF
        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        # 如果需要保存每一帧
        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}')

    # 打印日志信息
    def print_log(self):
        # 定义日志数据
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        # 过滤掉空值
        data = [d for d in data if exists(d[1])]
        # 将日志数据格式化为字符串
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        # 打印日志
        print(log)

    # 返回模型���件名
    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    # 初始化文件夹
    def init_folders(self):
        # 创建结果文件夹和模型文件夹
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    # 清空文件夹
    def clear(self):
        # 删除模型文件夹、结果文件夹、FID 文件夹和配置文件夹
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        # 初始化文件夹
        self.init_folders()

    # 保存模型
    def save(self, num):
        # 保存模型数据
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__,
            'G_scaler': self.G_scaler.state_dict(),
            'D_scaler': self.D_scaler.state_dict()
        }

        # 将数据保存到文件
        torch.save(save_data, self.model_name(num))
        # 写入配置文件
        self.write_config()

    # 加载模型
    def load(self, num=-1, print_version=True):
        # 加载配置文件
        self.load_config()

        name = num
        if num == -1:
            checkpoints = self.get_checkpoints()

            if not exists(checkpoints):
                return

            name = checkpoints[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        if print_version and 'version' in load_data and self.is_main:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e

        if 'G_scaler' in load_data:
            self.G_scaler.load_state_dict(load_data['G_scaler'])
        if 'D_scaler' in load_data:
            self.D_scaler.load_state_dict(load_data['D_scaler'])
    # 获取所有检查点文件的路径列表
    def get_checkpoints(self):
        # 使用列表推导式获取所有以'model_'开头的文件路径
        file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
        # 使用map函数和lambda表达式将文件路径转换为对应的数字编号，并按编号排序
        saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))

        # 如果没有找到任何检查点文件，则返回None
        if len(saved_nums) == 0:
            return None

        # 返回排序后的检查点编号列表
        return saved_nums
```