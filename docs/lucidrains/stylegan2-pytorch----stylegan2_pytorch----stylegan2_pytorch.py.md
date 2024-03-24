# `.\lucidrains\stylegan2-pytorch\stylegan2_pytorch\stylegan2_pytorch.py`

```py
# 导入必要的库
import os
import sys
import math
import fire
import json

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter2d

import torchvision
from torchvision import transforms
from stylegan2_pytorch.version import __version__
from stylegan2_pytorch.diff_augment import DiffAugment

from vector_quantize_pytorch import VectorQuantize

from PIL import Image
from pathlib import Path

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim

# 检查是否有可用的 CUDA 设备
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# 常量定义
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']

# 辅助类定义

# 自定义异常类
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

# 展平类
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

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
    def forward(self, x):
        return self.fn(x) + x

# 通道归一化类
class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# 预归一化类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))

# 维度置换类
class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss

# 模糊类
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)

# 注意力机制

# 深度卷积类
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

# 线性注意力类
class LinearAttention(nn.Module):
    # 初始化函数，设置注意力机制的参数
    def __init__(self, dim, dim_head = 64, heads = 8):
        # 调用父类的初始化函数
        super().__init__()
        # 计算缩放因子
        self.scale = dim_head ** -0.5
        # 设置头数
        self.heads = heads
        # 计算内部维度
        inner_dim = dim_head * heads

        # 使用 GELU 作为非线性激活函数
        self.nonlin = nn.GELU()
        # 创建输入到查询向量的卷积层
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        # 创建输入到键值对的卷积层
        self.to_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)
        # 创建输出的卷积层
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    # 前向传播函数
    def forward(self, fmap):
        # 获取头数和特征图的高度、宽度
        h, x, y = self.heads, *fmap.shape[-2:]
        # 计算查询、键、值
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        # 重排查询、键、值的维度
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        # 对查询进行 softmax 操作
        q = q.softmax(dim = -1)
        # 对键进行 softmax 操作
        k = k.softmax(dim = -2)

        # 缩放查询
        q = q * self.scale

        # 计算上下文信息
        context = einsum('b n d, b n e -> b d e', k, v)
        # 计算输出
        out = einsum('b n d, b d e -> b n e', q, context)
        # 重排输出的维度
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # 使用非线性激活函数
        out = self.nonlin(out)
        # 返回输出
        return self.to_out(out)
# 定义一个包含自注意力和前馈的函数，用于图像处理
attn_and_ff = lambda chan: nn.Sequential(*[
    # 使用残差连接将通道数作为参数传入预标准化和线性注意力模块中
    Residual(PreNorm(chan, LinearAttention(chan))),
    # 使用残差连接将通道数作为参数传入预标准化和卷积模块中
    Residual(PreNorm(chan, nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# 辅助函数

# 判断值是否存在
def exists(val):
    return val is not None

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

# 返回默认值
def default(value, d):
    return value if exists(value) else d

# 无限循环迭代器
def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# 将元素转换为列表
def cast_list(el):
    return el if isinstance(el, list) else [el]

# 判断张量是否为空
def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return not exists(t)

# 如果张量包含 NaN，则抛出异常
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

# 损失反向传播
def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# 梯度惩罚
def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size(), device=images.device),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# 计算潜在空间长��
def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape, device=device),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

# 生成噪声
def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

# 生成噪声列表
def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

# 生成混合噪声列表
def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

# 将潜在向量转换为 W
def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

# 生成图像噪声
def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda(device)

# Leaky ReLU 激活函数
def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

# 分块评估
def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

# 将样式定义转换为张量
def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)

# 设置模型参数是否需要梯度
def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

# Slerp 插值
def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    # 计算两个向量的夹角的余弦值
    omega = torch.acos((low_norm * high_norm).sum(1))
    # 计算夹角的正弦值
    so = torch.sin(omega)
    # 根据插值参数val计算插值结果
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    # 返回插值结果
    return res
# losses

# 生成 Hinge 损失函数，返回 fake 数据的均值
def gen_hinge_loss(fake, real):
    return fake.mean()

# Hinge 损失函数，计算 real 和 fake 数据的损失
def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

# 对偶对比损失函数，计算 real_logits 和 fake_logits 之间的损失
def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    # 重排维度
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    # 定义损失函数
    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    # 返回损失函数结果
    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

# dataset

# 将 RGB 图像转换为带透明度的图像
def convert_rgb_to_transparent(image):
    if image.mode != 'RGBA':
        return image.convert('RGBA')
    return image

# 将带透明度的图像转换为 RGB 图像
def convert_transparent_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

# 扩展灰度图像类
class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

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

        if not exists(alpha) and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))

# 调整图像大小至最小尺寸
def resize_to_minimum_size(min_size, image):
    if max(*image.size) < min_size:
        return torchvision.transforms.functional.resize(image, min_size)
    return image

# 数据集类
class Dataset(data.Dataset):
    def __init__(self, folder, image_size, transparent = False, aug_prob = 0.):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        assert len(self.paths) > 0, f'No images were found in {folder} for training'

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(transparent))
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# augmentations

# 随机水平翻转图像
def random_hflip(tensor, prob):
    if prob < random():
        return tensor
    return torch.flip(tensor, dims=(3,))

# 增强包装类
class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob = 0., types = [], detach = False):
        if random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)

# stylegan2 classes

# 等权线性层
class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul
    # 定义一个前向传播函数，接收输入并返回线性变换的结果
    def forward(self, input):
        # 使用线性变换函数对输入进行处理，其中权重乘以学习率倍数，偏置也乘以学习率倍数
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)
class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)



# 定义一个风格向量化器模块，用于将输入向量进行风格化处理
class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        # 初始化函数
        super().__init__()

        # 创建一个空的层列表
        layers = []
        # 根据深度循环创建一组相同结构的层
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        # 将层列表组合成一个序列
        self.net = nn.Sequential(*layers)

    # 前向传播函数
    def forward(self, x):
        # 对输入进行归一化处理
        x = F.normalize(x, dim=1)
        return self.net(x)



class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if exists(prev_rgb):
            x = x + prev_rgb

        if exists(self.upsample):
            x = self.upsample(x)

        return x



# 定义一个 RGB 模块，用于处理 RGB 图像数据
class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        # 初始化函数
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        # 根据是否包含 alpha 通道确定输出通道数
        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        # 根据是否需要上采样创建上采样模块
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False),
            Blur()
        ) if upsample else None

    # 前向传播函数
    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        # 如果存在前一层 RGB 数据，则进行相加操作
        if exists(prev_rgb):
            x = x + prev_rgb

        # 如果存在上采样模块，则进行上采样操作
        if exists(self.upsample):
            x = self.upsample(x)

        return x



class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x



# 定义一个带有调制的卷积模块
class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        # 初始化函数
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    # 计算填充大小的函数
    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    # 前向传播函数
    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x



class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if exists(self.upsample):
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb



# 定义一个生成器模块
class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        # 初始化函数
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    # 前向传播函数
    def forward(self, x, prev_rgb, istyle, inoise):
        # 如果需要上采样，则进行上采样操作
        if exists(self.upsample):
            x = self.upsample(x)

        # 裁剪噪声数据
        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb



class DiscriminatorBlock(nn.Module):



# 定义一个鉴别器模块
class DiscriminatorBlock(nn.Module):
    # 初始化函数，定义了一个卷积层 conv_res，用于降采样
    def __init__(self, input_channels, filters, downsample=True):
        # 调用父类的初始化函数
        super().__init__()
        # 定义一个卷积层 conv_res，用于降采样，1x1卷积核，stride为2（如果 downsample 为 True）
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride = (2 if downsample else 1))

        # 定义一个神经网络模型 net，包含两个卷积层和激活函数
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        # 如果 downsample 为 True，则定义一个下采样模块 downsample，包含模糊层和卷积层
        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(filters, filters, 3, padding = 1, stride = 2)
        ) if downsample else None

    # 前向传播函数，接收输入 x，返回处理后的结果
    def forward(self, x):
        # 对输入 x 进行卷积操作，得到 res
        res = self.conv_res(x)
        # 对输入 x 进行神经网络模型 net 的处理
        x = self.net(x)
        # 如果 downsample 存在，则对 x 进行下采样
        if exists(self.downsample):
            x = self.downsample(x)
        # 将下采样后的 x 与 res 相加，并乘以 1/sqrt(2)
        x = (x + res) * (1 / math.sqrt(2))
        # 返回处理后的结果 x
        return x
class Generator(nn.Module):
    # 生成器类，继承自 nn.Module
    def __init__(self, image_size, latent_dim, network_capacity = 16, transparent = False, attn_layers = [], no_const = False, fmap_max = 512):
        # 初始化函数，接受图像大小、潜在维度、网络容量、是否透明、注意力层等参数
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last,
                rgba = transparent
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        # 前向传播函数，接受样式和输入噪声
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb

class Discriminator(nn.Module):
    # 判别器类，继承自 nn.Module
    def __init__(self, image_size, network_capacity = 16, fq_layers = [], fq_dict_size = 256, attn_layers = [], transparent = False, fmap_max = 512):
        # 初始化函数，接受图像大小、网络容量、fq_layers、fq_dict_size、attn_layers、是否透明、fmap_max等参数
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_chan, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 获取输入 x 的 batch size
        b, *_ = x.shape

        # 初始化量化损失为零张量，与输入 x 相同的设备
        quantize_loss = torch.zeros(1).to(x)

        # 遍历每个块，注意力块和量化块
        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            # 对输入 x 应用块操作
            x = block(x)

            # 如果存在注意力块，则对输入 x 应用注意力块
            if exists(attn_block):
                x = attn_block(x)

            # 如果存在量化块，则对输入 x 应用量化块，并计算损失
            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        # 对最终输出 x 应用最终卷积层
        x = self.final_conv(x)
        # 将输出 x 展平
        x = self.flatten(x)
        # 将展平后的输出 x 转换为 logit
        x = self.to_logit(x)
        # 压缩输出 x 的维度，去除大小为 1 的维度
        return x.squeeze(), quantize_loss
class StyleGAN2(nn.Module):
    # 定义 StyleGAN2 类，继承自 nn.Module
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, cl_reg = False, steps = 1, lr = 1e-4, ttur_mult = 2, fq_layers = [], fq_dict_size = 256, attn_layers = [], no_const = False, lr_mlp = 0.1, rank = 0):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数
        self.lr = lr
        self.steps = steps
        # 设置学习率和训练步数
        self.ema_updater = EMA(0.995)
        # 创建指数移动平均对象

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, fq_layers = fq_layers, fq_dict_size = fq_dict_size, attn_layers = attn_layers, transparent = transparent, fmap_max = fmap_max)
        # 创建 StyleVectorizer、Generator 和 Discriminator 对象

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mlp)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, attn_layers = attn_layers, no_const = no_const)
        # 创建 StyleVectorizer 和 Generator 对象

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner
            # 导入 ContrastiveLearner 类
            assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
            # 断言透明度为假
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')
            # 创建 ContrastiveLearner 对象

        self.D_aug = AugWrapper(self.D, image_size)
        # 创建 AugWrapper 对象

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)
        # 设置 StyleVectorizer 和 Generator 的梯度计算为 False

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))
        # 初始化生成器和判别器的优化器

        self._init_weights()
        self.reset_parameter_averaging()
        # 初始化权重和参数平均

        self.cuda(rank)
        # 将模型移动到 GPU

        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)
            # 使用混合精度训练

    def _init_weights(self):
        # 初始化权重函数
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # 使用 kaiming_normal_ 初始化权重

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)
        # 初始化 Generator 中的权重

    def EMA(self):
        # 定义指数移动平均函数
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)
        # 更新指数移动平均参数

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)
        # 更新 StyleVectorizer 和 Generator 的指数移动平均参数

    def reset_parameter_averaging(self):
        # 重置参数平均函数
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())
        # 加载当前状态到指数移动平均模型

    def forward(self, x):
        # 前向传播函数
        return x
        # 返回输入

class Trainer():
    # 定义 Trainer 类
    # 初始化函数，设置各种参数和默认值
    def __init__(
        self,
        name = 'default',  # 模型名称，默认为'default'
        results_dir = 'results',  # 结果保存目录，默认为'results'
        models_dir = 'models',  # 模型保存目录，默认为'models'
        base_dir = './',  # 基础目录，默认为当前目录
        image_size = 128,  # 图像大小，默认为128
        network_capacity = 16,  # 网络容量，默认为16
        fmap_max = 512,  # 特征图最大值，默认为512
        transparent = False,  # 是否透明，默认为False
        batch_size = 4,  # 批量大小，默认为4
        mixed_prob = 0.9,  # 混合概率，默认为0.9
        gradient_accumulate_every=1,  # 梯度累积步数，默认为1
        lr = 2e-4,  # 学习率，默认为2e-4
        lr_mlp = 0.1,  # MLP学习率，默认为0.1
        ttur_mult = 2,  # TTUR倍数，默认为2
        rel_disc_loss = False,  # 相对鉴别器损失，默认为False
        num_workers = None,  # 工作进程数，默认为None
        save_every = 1000,  # 保存频率，默认为1000
        evaluate_every = 1000,  # 评估频率，默认为1000
        num_image_tiles = 8,  # 图像平铺数，默认为8
        trunc_psi = 0.6,  # 截断值，默认为0.6
        fp16 = False,  # 是否使用FP16，默认为False
        cl_reg = False,  # 是否使用对比损失正则化，默认为False
        no_pl_reg = False,  # 是否不使用PL正则化，默认为False
        fq_layers = [],  # FQ层列表，默认为空列表
        fq_dict_size = 256,  # FQ字典大小，默认为256
        attn_layers = [],  # 注意力层列表，默认为空列表
        no_const = False,  # 是否不使用常数，默认为False
        aug_prob = 0.,  # 数据增强概率，默认为0
        aug_types = ['translation', 'cutout'],  # 数据增强类型，默认为['translation', 'cutout']
        top_k_training = False,  # 是否使用Top-K训练，默认为False
        generator_top_k_gamma = 0.99,  # 生成器Top-K Gamma值，默认为0.99
        generator_top_k_frac = 0.5,  # 生成器Top-K分数，默认为0.5
        dual_contrast_loss = False,  # 是否使用双对比损失，默认为False
        dataset_aug_prob = 0.,  # 数据集增强概率，默认为0
        calculate_fid_every = None,  # 计算FID频率，默认为None
        calculate_fid_num_images = 12800,  # 计算FID图像数，默认为12800
        clear_fid_cache = False,  # 是否清除FID缓存，默认为False
        is_ddp = False,  # 是否使用DDP，默认为False
        rank = 0,  # 排名，默认为0
        world_size = 1,  # 世界大小，默认为1
        log = False,  # 是否记��日志，默认为False
        *args,  # 可变位置参数
        **kwargs  # 可变关键字参数
    ):
        self.GAN_params = [args, kwargs]  # GAN参数列表
        self.GAN = None  # GAN对象

        self.name = name  # 设置模型名称

        base_dir = Path(base_dir)  # 将基础目录转换为Path对象
        self.base_dir = base_dir  # 设置基础目录
        self.results_dir = base_dir / results_dir  # 设置结果保存目录
        self.models_dir = base_dir / models_dir  # 设置模型保存目录
        self.fid_dir = base_dir / 'fid' / name  # 设置FID目录
        self.config_path = self.models_dir / name / '.config.json'  # 设置配置文件路径

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'  # 断言图像大小必须是2的幂次方
        self.image_size = image_size  # 设置图像大小
        self.network_capacity = network_capacity  # 设置网络容量
        self.fmap_max = fmap_max  # 设置特征图最大值
        self.transparent = transparent  # 设置是否透明

        self.fq_layers = cast_list(fq_layers)  # 将FQ层转换为列表
        self.fq_dict_size = fq_dict_size  # 设置FQ字典大小
        self.has_fq = len(self.fq_layers) > 0  # 判断是否有FQ层

        self.attn_layers = cast_list(attn_layers)  # 将注意力层转换为列表
        self.no_const = no_const  # 设置是否不使用常数

        self.aug_prob = aug_prob  # 设置数据增强概率
        self.aug_types = aug_types  # 设置数据增强类型

        self.lr = lr  # 设置学习率
        self.lr_mlp = lr_mlp  # 设置MLP学习率
        self.ttur_mult = ttur_mult  # 设置TTUR倍数
        self.rel_disc_loss = rel_disc_loss  # 设置是否相对鉴别器损失
        self.batch_size = batch_size  # 设置批量大小
        self.num_workers = num_workers  # 设置工作进程数
        self.mixed_prob = mixed_prob  # 设置混合概率

        self.num_image_tiles = num_image_tiles  # 设置图像平铺数
        self.evaluate_every = evaluate_every  # 设置评估频率
        self.save_every = save_every  # 设置保存频率
        self.steps = 0  # 步数初始化为0

        self.av = None  # 初始化av
        self.trunc_psi = trunc_psi  # 设置截断值

        self.no_pl_reg = no_pl_reg  # 设置是否不使用PL正则化
        self.pl_mean = None  # 初始化PL均值

        self.gradient_accumulate_every = gradient_accumulate_every  # 设置梯度累积步数

        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'  # 断言Apex是否可用
        self.fp16 = fp16  # 设置是否使用FP16

        self.cl_reg = cl_reg  # 设置是否使用对比损失正则化

        self.d_loss = 0  # 初始化鉴别器损失
        self.g_loss = 0  # 初始化生成器损失
        self.q_loss = None  # 初始化Q损失
        self.last_gp_loss = None  # 初始化上一次梯度惩罚损失
        self.last_cr_loss = None  # 初始化上一次对比损失
        self.last_fid = None  # 初始化上一次FID

        self.pl_length_ma = EMA(0.99)  # 初始化PL长度移动平均
        self.init_folders()  # 初始化文件夹

        self.loader = None  # 初始化数据加载器
        self.dataset_aug_prob = dataset_aug_prob  # 设置数据集增强概率

        self.calculate_fid_every = calculate_fid_every  # 设置计算FID频率
        self.calculate_fid_num_images = calculate_fid_num_images  # 设置计算FID图像数
        self.clear_fid_cache = clear_fid_cache  # 设置是否清除FID缓存

        self.top_k_training = top_k_training  # 设置是否使用Top-K训练
        self.generator_top_k_gamma = generator_top_k_gamma  # 设置生成器Top-K Gamma值
        self.generator_top_k_frac = generator_top_k_frac  # 设置生成器Top-K分数

        self.dual_contrast_loss = dual_contrast_loss  # 设置是否使用双对比损失

        assert not (is_ddp and cl_reg), 'Contrastive loss regularization does not work well with multi GPUs yet'  # 断言对比损失正则化在多GPU上不起作用
        self.is_ddp = is_ddp  # 设置是否使用DDP
        self.is_main = rank == 0  # 判断是否为主进程
        self.rank = rank  # 设置排名
        self.world_size = world_size  # 设置世界大小

        self.logger = aim.Session(experiment=name) if log else None  # 设置记录器
    @property
    # 返回图片的扩展名，如果是透明图片则返回png，否则返回jpg
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    # 返回检查点编号，根据步数和保存频率计算得出
    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)

    # 返回超参数字典，包括图片大小和网络容量
    @property
    def hparams(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}
        
    # 初始化生成对抗网络
    def init_GAN(self):
        args, kwargs = self.GAN_params
        # 创建StyleGAN2对象
        self.GAN = StyleGAN2(lr = self.lr, lr_mlp = self.lr_mlp, ttur_mult = self.ttur_mult, image_size = self.image_size, network_capacity = self.network_capacity, fmap_max = self.fmap_max, transparent = self.transparent, fq_layers = self.fq_layers, fq_dict_size = self.fq_dict_size, attn_layers = self.attn_layers, fp16 = self.fp16, cl_reg = self.cl_reg, no_const = self.no_const, rank = self.rank, *args, **kwargs)

        # 如果是分布式训练，使用DDP包装GAN的各个部分
        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank]}
            self.S_ddp = DDP(self.GAN.S, **ddp_kwargs)
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

        # 如果存在日志记录器，设置参数
        if exists(self.logger):
            self.logger.set_params(self.hparams)

    # 写入配置信息到配置文件
    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    # 从配置文件加载配置信息
    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        # 更新配置信息
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.fq_layers = config['fq_layers']
        self.fq_dict_size = config['fq_dict_size']
        self.fmap_max = config.pop('fmap_max', 512)
        self.attn_layers = config.pop('attn_layers', [])
        self.no_const = config.pop('no_const', False)
        self.lr_mlp = config.pop('lr_mlp', 0.1)
        del self.GAN
        self.init_GAN()

    # 返回配置信息字典
    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'lr_mlp': self.lr_mlp, 'transparent': self.transparent, 'fq_layers': self.fq_layers, 'fq_dict_size': self.fq_dict_size, 'attn_layers': self.attn_layers, 'no_const': self.no_const}

    # 设置数据源
    def set_data_src(self, folder):
        # 创建数据集对象
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        num_workers = num_workers = default(self.num_workers, NUM_CORES if not self.is_ddp else 0)
        # 创建数据加载器
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        dataloader = data.DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # 如果数据集较小，自动设置数据增强概率
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    # 禁用梯度计算
    @torch.no_grad()
    # 定义一个评估函数，用于生成图像并计算 FID 分数
    def evaluate(self, num = 0, trunc = 1.0):
        # 将 GAN 设置为评估模式
        self.GAN.eval()
        # 获取图像文件扩展名和图像瓦片数量
        ext = self.image_extension
        num_rows = self.num_image_tiles

        # 获取潜在空间维度、图像尺寸和层数
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # 生成潜在向量和噪声
        latents = noise_list(num_rows ** 2, num_layers, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        # 生成正常图像
        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # 生成移动平均图像
        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        # 生成混合图像
        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda(self.rank)
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim, device=self.rank)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    # 计算 FID 分数
    @torch.no_grad()
    def calculate_fid(self, num_batches):
        # 导入 FID 分数计算模块并清空 GPU 缓存
        from pytorch_fid import fid_score
        torch.cuda.empty_cache()

        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # 删除用于 FID 计算的现有文件并重新创建目录
        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    filename = str(k + batch_num * self.batch_size)
                    torchvision.utils.save_image(image, str(real_path / f'{filename}.png'))

        # 生成一堆假图像在 results / name / fid_fake 目录下
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # 生成潜在向量和噪声
            latents = noise_list(self.batch_size, num_layers, latent_dim, device=self.rank)
            noise = image_noise(self.batch_size, image_size, device=self.rank)

            # 生成移动平均图像
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, noise, trunc_psi = self.trunc_psi)

            for j, image in enumerate(generated_images.unbind(0)):
                torchvision.utils.save_image(image, str(fake_path / f'{str(j + batch_num * self.batch_size)}-ema.{ext}'))

        # 返回 FID 分数
        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, noise.device, 2048)

    @torch.no_grad()
    # 对输入的张量进行截断操作，将其限制在一定范围内
    def truncate_style(self, tensor, trunc_psi = 0.75):
        # 获取模型的尺寸、批量大小和潜在维度
        S = self.GAN.S
        batch_size = self.batch_size
        latent_dim = self.GAN.G.latent_dim

        # 如果平均向量不存在，则生成一个
        if not exists(self.av):
            z = noise(2000, latent_dim, device=self.rank)
            samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)

        # 将平均向量转换为 PyTorch 张量
        av_torch = torch.from_numpy(self.av).cuda(self.rank)
        # 对输入张量进行截断操作
        tensor = trunc_psi * (tensor - av_torch) + av_torch
        return tensor

    # 对样式进行截断操作
    @torch.no_grad()
    def truncate_style_defs(self, w, trunc_psi = 0.75):
        w_space = []
        for tensor, num_layers in w:
            tensor = self.truncate_style(tensor, trunc_psi = trunc_psi)            
            w_space.append((tensor, num_layers))
        return w_space

    # 生成经过截断的图像
    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        w = map(lambda t: (S(t[0]), t[1]), style)
        w_truncated = self.truncate_style_defs(w, trunc_psi = trunc_psi)
        w_styles = styles_def_to_tensor(w_truncated)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    # 生成插值图像
    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, num_steps = 100, save_frames = False):
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # 生成潜在向量和噪声
        latents_low = noise(num_rows ** 2, latent_dim, device=self.rank)
        latents_high = noise(num_rows ** 2, latent_dim, device=self.rank)
        n = image_noise(num_rows ** 2, image_size, device=self.rank)

        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            # 如果需要透明背景，则进行处理
            if self.transparent:
                background = Image.new("RGBA", pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        # 保存生成的插值图像为 GIF
        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        # 如果需要保存每一帧图像，则保存为单独的文件
        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}')

    # 打印日志信息
    def print_log(self):
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('PL', self.pl_mean),
            ('CR', self.last_cr_loss),
            ('Q', self.q_loss),
            ('FID', self.last_fid)
        ]

        data = [d for d in data if exists(d[1])]
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        print(log)

    # 记录日志信息
    def track(self, value, name):
        if not exists(self.logger):
            return
        self.logger.track(value, name = name)

    # 返回模型文件名
    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    # 初始化文件夹
    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)
    # 清空模型、结果和文件夹目录
    def clear(self):
        # 删除模型目录下的所有文件和文件夹
        rmtree(str(self.models_dir / self.name), True)
        # 删除结果目录下的所有文件和文件夹
        rmtree(str(self.results_dir / self.name), True)
        # 删除 FID 目录下的所有文件和文件夹
        rmtree(str(self.fid_dir), True)
        # 删除配置文件路径
        rmtree(str(self.config_path), True)
        # 初始化文件夹
        self.init_folders()

    # 保存模型
    def save(self, num):
        # 保存模型的数据和版本号
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__
        }

        # 如果模型使用了混合精度训练，保存混合精度训练的状态
        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        # 将保存的数据保存到模型文件中
        torch.save(save_data, self.model_name(num))
        # 写入配置文件
        self.write_config()

    # 加载模型
    def load(self, num = -1):
        # 加载配置文件
        self.load_config()

        # 如果未指定加载的模型编号，则查找最新的模型文件
        name = num
        if num == -1:
            # 获取模型目录下所有以'model_'开头的文件路径
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            # 提取文件名中的数字部分并排序
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            # 如果没有保存的模型文件，则直接返回
            if len(saved_nums) == 0:
                return
            # 获取最新的模型编号
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        # 计算当前步数
        self.steps = name * self.save_every

        # 加载模型数据
        load_data = torch.load(self.model_name(name))

        # 打印加载的模型版本号
        if 'version' in load_data:
            print(f"loading from version {load_data['version']}")

        # 尝试加载 GAN 模型的状态字典
        try:
            self.GAN.load_state_dict(load_data['GAN'])
        except Exception as e:
            # 加载失败时提示用户尝试降级软件包版本
            print('unable to load save model. please try downgrading the package to the version specified by the saved model')
            raise e
        # 如果使用了混合精度训练且保存了混合精度训练的状态，则加��
        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])
# 定义一个模型加载器类
class ModelLoader:
    # 初始化方法，接收基本目录、名称和加载位置参数
    def __init__(self, *, base_dir, name = 'default', load_from = -1):
        # 创建一个Trainer对象作为模型属性
        self.model = Trainer(name = name, base_dir = base_dir)
        # 加载模型
        self.model.load(load_from)

    # 将噪声转换为样式向量的方法
    def noise_to_styles(self, noise, trunc_psi = None):
        # 将噪声数据移到GPU上
        noise = noise.cuda()
        # 通过SE模块将噪声转换为样式向量
        w = self.model.GAN.SE(noise)
        # 如果截断参数存在，则对样式向量进行截断
        if exists(trunc_psi):
            w = self.model.truncate_style(w)
        return w

    # 将样式向量转换为图像的方法
    def styles_to_images(self, w):
        # 获取样式向量的形状信息
        batch_size, *_ = w.shape
        # 获取生成器的层数和图像大小
        num_layers = self.model.GAN.GE.num_layers
        image_size = self.model.image_size
        # 构建样式向量定义列表
        w_def = [(w, num_layers)]

        # 将样式向量定义列表转换为张量
        w_tensors = styles_def_to_tensor(w_def)
        # 生成图像所需的噪声数据
        noise = image_noise(batch_size, image_size, device = 0)

        # 通过GE模块生成图像
        images = self.model.GAN.GE(w_tensors, noise)
        # 将图像像素值限制在0到1之间
        images.clamp_(0., 1.)
        return images
```