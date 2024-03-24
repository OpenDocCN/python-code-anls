# `.\lucidrains\unet-stylegan2\unet_stylegan2\unet_stylegan2.py`

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

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from torch.optim import Adam
from torch.autograd import grad as torch_grad

import torchvision
from torchvision import transforms

from linear_attention_transformer import ImageLinearAttention

from PIL import Image
from pathlib import Path

# 尝试导入 apex 库，设置 APEX_AVAILABLE 变量
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# 检查是否有可用的 CUDA 设备
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# 获取 CPU 核心数量
num_cores = multiprocessing.cpu_count()

# 常量定义

# 支持的图片文件格式
EXTS = ['jpg', 'jpeg', 'png', 'webp']
# 微小的常数，用于避免除零错误
EPS = 1e-8

# 辅助类定义

# 自定义异常类，用于处理 NaN 异常
class NanException(Exception):
    pass

# 指数移动平均类
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# 随机应用类，根据概率应用不同的函数
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

# 展平类
class Flatten(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
    def forward(self, x):
        return x.flatten(self.index)

# Rezero 类
class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.fn(x) * self.g

# 图像的自注意力和前馈网络层
attn_and_ff = lambda chan: nn.Sequential(*[
    Residual(Rezero(ImageLinearAttention(chan, norm_queries = True))),
    Residual(Rezero(nn.Sequential(nn.Conv2d(chan, chan * 2, 1), leaky_relu(), nn.Conv2d(chan * 2, chan, 1))))
])

# 辅助函数定义

# 返回默认值
def default(value, d):
    return d if value is None else value

# 无限循环迭代器
def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# 将元素转换为列表
def cast_list(el):
    return el if isinstance(el, list) else [el]

# 检查张量是否为空
def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return t is None

# 如果张量包含 NaN，则抛出异常
def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

# 反向传播函数，支持混合精度训练
def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# 计算梯度惩罚项
def gradient_penalty(images, outputs, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=outputs, inputs=images,
                           grad_outputs=list(map(lambda t: torch.ones(t.size()).cuda(), outputs)),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# 计算潜在空间长度
def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape).cuda(),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

# 生成随机噪声
def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cuda()

# 生成多层随机噪声列表
def noise_list(n, layers, latent_dim):
    # 返回一个包含噪声和层信息的元组列表
    return [(noise(n, latent_dim), layers)]
# 生成一个混合的噪声列表，包含两个噪声列表的和
def mixed_list(n, layers, latent_dim):
    # 随机选择一个整数作为分割点
    tt = int(torch.rand(()).numpy() * layers)
    # 返回两个噪声列表的和
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)

# 将潜在向量描述转换为样式向量和层数的元组列表
def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

# 生成一个指定大小的图像噪声
def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda()

# 返回一个带有泄漏整流的激活函数
def leaky_relu(p=0.2):
    return nn.LeakyReLU(p)

# 将输入参数按照最大批量大小分块，对模型进行评估
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

# Slerp 插值函数
def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# 热身函数，用于在一定步数内线性增加数值
def warmup(start, end, max_steps, current_step):
    if current_step > max_steps:
        return end
    return (end - start) * (current_step / max_steps) + start

# 对张量进行对数运算
def log(t, eps = 1e-6):
    return torch.log(t + eps)

# 生成 CutMix 的坐标
def cutmix_coordinates(height, width, alpha = 1.):
    lam = np.random.beta(alpha, alpha)

    cx = np.random.uniform(0, width)
    cy = np.random.uniform(0, height)
    w = width * np.sqrt(1 - lam)
    h = height * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, width)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, height)))

    return ((y0, y1), (x0, x1)), lam

# 执行 CutMix 操作
def cutmix(source, target, coors, alpha = 1.):
    source, target = map(torch.clone, (source, target))
    ((y0, y1), (x0, x1)), _ = coors
    source[:, :, y0:y1, x0:x1] = target[:, :, y0:y1, x0:x1]
    return source

# 对源和目标进行遮罩操作
def mask_src_tgt(source, target, mask):
    return source * mask + (1 - mask) * target

# 数据集

# 将 RGB 图像转换为带透明通道的图像
def convert_rgb_to_transparent(image):
    if image.mode == 'RGB':
        return image.convert('RGBA')
    return image

# 将带透明通道的图像转换为 RGB 图像
def convert_transparent_to_rgb(image):
    if image.mode == 'RGBA':
        return image.convert('RGB')
    return image

# 扩展灰度图像通道数
class expand_greyscale(object):
    def __init__(self, num_channels):
        self.num_channels = num_channels
    def __call__(self, tensor):
        return tensor.expand(self.num_channels, -1, -1)

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

        convert_image_fn = convert_transparent_to_rgb if not transparent else convert_rgb_to_transparent
        num_channels = 3 if not transparent else 4

        self.transform = transforms.Compose([
            transforms.Lambda(convert_image_fn),
            transforms.Lambda(partial(resize_to_minimum_size, image_size)),
            transforms.Resize(image_size),
            RandomApply(aug_prob, transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), ratio=(0.98, 1.02)), transforms.CenterCrop(image_size)),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale(num_channels))
        ])

    def __len__(self):
        return len(self.paths)
    # 定义一个特殊方法，用于获取对象中指定索引位置的元素
    def __getitem__(self, index):
        # 获取指定索引位置的路径
        path = self.paths[index]
        # 打开指定路径的图像文件
        img = Image.open(path)
        # 对图像进行变换处理并返回
        return self.transform(img)
# 定义一个生成器块类
class GeneratorBlock(nn.Module):
    # 初始化函数
    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        # 将输入的潜在向量映射到输入通道数
        self.to_style = nn.Linear(latent_dim, input_channel)
        
        # 如果是 RGBA 模式，则输出通道数为 4，否则为 3
        out_filters = 3 if not rgba else 4
        # 定义卷积层，不进行调制
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)
        
        # 如果需要上采样，则定义上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

    # 前向传播函数
    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        # 将潜在向量映射到输入通道数
        style = self.to_style(istyle)
        # 使用卷积层进行特征提取
        x = self.conv(x, style)

        # 如果有上一个 RGB 图像，则进行残差连接
        if prev_rgb is not None:
            x = x + prev_rgb

        # 如果需要上采样，则进行上采样操作
        if self.upsample is not None:
            x = self.upsample(x)

        return x
    # 初始化函数，定义生成器的结构
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        # 调用父类的初始化函数
        super().__init__()
        # 如果需要上采样，则创建上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        # 创建将潜在向量映射到输入通道的全连接层
        self.to_style1 = nn.Linear(latent_dim, input_channels)
        # 创建将噪声映射到滤波器数量的全连接层
        self.to_noise1 = nn.Linear(1, filters)
        # 创建卷积层，使用自定义的Conv2DMod类
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        # 创建将潜在向量映射到滤波器数量的全连接层
        self.to_style2 = nn.Linear(latent_dim, filters)
        # 创建将噪声映射到滤波器数量的全连接层
        self.to_noise2 = nn.Linear(1, filters)
        # 创建卷积层，使用自定义的Conv2DMod类
        self.conv2 = Conv2DMod(filters, filters, 3)

        # 定义激活函数为LeakyReLU
        self.activation = leaky_relu()
        # 创建RGBBlock实例，用于生成RGB输出
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    # 前向传播函数，定义生成器的前向传播过程
    def forward(self, x, prev_rgb, istyle, inoise):
        # 如果需要上采样，则对输入进行上采样
        if self.upsample is not None:
            x = self.upsample(x)

        # 裁剪噪声张量，使其与输入张量的尺寸相匹配
        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        # 将噪声映射到滤波器数量，并进行维度变换
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        # 将潜在向量映射到输入通道，并进行卷积操作
        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        # 将潜在向量映射到滤波器数量，并进行卷积操作
        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        # 生成RGB输出
        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb
# 定义一个包含两个卷积层和激活函数的序列模块
def double_conv(chan_in, chan_out):
    return nn.Sequential(
        nn.Conv2d(chan_in, chan_out, 3, padding=1),  # 3x3卷积层，输入通道数为chan_in，输出通道数为chan_out，填充为1
        leaky_relu(),  # 使用LeakyReLU激活函数
        nn.Conv2d(chan_out, chan_out, 3, padding=1),  # 3x3卷积层，输入通道数为chan_out，输出通道数为chan_out，填充为1
        leaky_relu()  # 使用LeakyReLU激活函数
    )

# 定义一个下采样块模块
class DownBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))  # 1x1卷积层，输入通道数为input_channels，输出通道数为filters，步长为2或1

        self.net = double_conv(input_channels, filters)  # 使用double_conv函数创建卷积层序列
        self.down = nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None  # 下采样卷积层，输入通道数为filters，输出通道数为filters，填充为1，步长为2或None

    def forward(self, x):
        res = self.conv_res(x)  # 对输入x进行1x1卷积
        x = self.net(x)  # 使用卷积层序列处理输入x
        unet_res = x

        if self.down is not None:
            x = self.down(x)  # 如果存在下采样卷积层，则对x进行下采样

        x = x + res  # 将1x1卷积结果与处理后的x相加
        return x, unet_res

# 定义一个上采样块模块
class UpBlock(nn.Module):
    def __init__(self, input_channels, filters):
        super().__init__()
        self.conv_res = nn.ConvTranspose2d(input_channels // 2, filters, 1, stride=2)  # 转置卷积层，输入通道数为input_channels的一半，输出通道数为filters，步长为2
        self.net = double_conv(input_channels, filters)  # 使用double_conv函数创建卷积层序列
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样层，尺度因��为2，插值模式为双���性插值，不对齐角点

    def forward(self, x, res):
        *_, h, w = x.shape
        conv_res = self.conv_res(x, output_size=(h * 2, w * 2))  # 对输入x进行转置卷积
        x = self.up(x)  # 对输入x进行上采样
        x = torch.cat((x, res), dim=1)  # 在通道维度上拼接x和res
        x = self.net(x)  # 使用卷积层序列处理拼接后的x
        x = x + conv_res  # 将转置卷积结果与处理后的x相加
        return x

# 定义一个生成器模块
class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=16, transparent=False, no_const=False, fmap_max=512):
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
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, init_channels, 4, 1, 0, bias=False)  # 转置卷积层，输入通道数为latent_dim，输出通道数为init_channels，核大小为4，步长为1，填充为0，无偏置
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))  # 初始化块参数为随机张量

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)  # 3x3卷积层，输入通道数为filters[0]，输出通道数为filters[0]，填充为1

        self.blocks = nn.ModuleList([])  # 创建模块列表
        self.attns = nn.ModuleList([])  # 创建模块列表

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan)  # 获取注意力函数
            self.attns.append(attn_fn)  # 添加到注意力模块列表

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent
            )
            self.blocks.append(block)  # 添加生成器块模块到模块列表

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]
        image_size = self.image_size

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)  # 使用平均风格向量生成初始块
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)  # 扩展初始块参数

        x = self.initial_conv(x)  # 对初始块进行卷积
        styles = styles.transpose(0, 1)  # 转置风格张量

        rgb = None
        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)  # 如果存在注意力模块，则应用注意力
            x, rgb = block(x, rgb, style, input_noise)  # 使用生成器块模块处理x和rgb

        return rgb  # 返回rgb

class Discriminator(nn.Module):
    # 初始化函数，设置神经网络的参数
    def __init__(self, image_size, network_capacity = 16, transparent = False, fmap_max = 512):
        # 调用父类的初始化函数
        super().__init__()
        # 计算网络层数
        num_layers = int(log2(image_size) - 3)
        # 初始化滤波器数量
        num_init_filters = 3 if not transparent else 4

        blocks = []
        # 计算每一层的滤波器数量
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]

        # 设置最大滤波器数量
        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        filters[-1] = filters[-2]

        # 组合输入输出通道数
        chan_in_out = list(zip(filters[:-1], filters[1:]))
        chan_in_out = list(map(list, chan_in_out))

        down_blocks = []
        attn_blocks = []

        # 遍历每一层，创建下采样块和注意力块
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DownBlock(in_chan, out_chan, downsample = is_not_last)
            down_blocks.append(block)

            attn_fn = attn_and_ff(out_chan)
            attn_blocks.append(attn_fn)

        # 将下采样块和注意力块转换为 ModuleList
        self.down_blocks = nn.ModuleList(down_blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)

        last_chan = filters[-1]

        # 定义输出层
        self.to_logit = nn.Sequential(
            leaky_relu(),
            nn.AvgPool2d(image_size // (2 ** num_layers)),
            Flatten(1),
            nn.Linear(last_chan, 1)
        )

        self.conv = double_conv(last_chan, last_chan)

        # 反向遍历通道输入输出，创建上采样块
        dec_chan_in_out = chan_in_out[:-1][::-1]
        self.up_blocks = nn.ModuleList(list(map(lambda c: UpBlock(c[1] * 2, c[0]), dec_chan_in_out)))
        self.conv_out = nn.Conv2d(3, 1, 1)

    # 前向传播函数
    def forward(self, x):
        b, *_ = x.shape

        residuals = []

        # 遍历下采样块和注意力块
        for (down_block, attn_block) in zip(self.down_blocks, self.attn_blocks):
            x, unet_res = down_block(x)
            residuals.append(unet_res)

            if attn_block is not None:
                x = attn_block(x)

        x = self.conv(x) + x
        enc_out = self.to_logit(x)

        # 反向遍历上采样块，生成解码输出
        for (up_block, res) in zip(self.up_blocks, residuals[:-1][::-1]):
            x = up_block(x, res)

        dec_out = self.conv_out(x)
        return enc_out.squeeze(), dec_out
class StyleGAN2(nn.Module):
    # 定义 StyleGAN2 类，继承自 nn.Module
    def __init__(self, image_size, latent_dim = 512, fmap_max = 512, style_depth = 8, network_capacity = 16, transparent = False, fp16 = False, steps = 1, lr = 1e-4, ttur_mult = 2, no_const = False, lr_mul = 0.1, aug_types = ['translation', 'cutout']):
        # 初始化函数，接受多个参数
        super().__init__()
        # 调用父类的初始化函数

        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)
        # 设置学习率、步数和指数移动平均更新器

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mul)
        self.G = Generator(image_size, latent_dim, network_capacity, transparent = transparent, no_const = no_const, fmap_max = fmap_max)
        self.D = Discriminator(image_size, network_capacity, transparent = transparent, fmap_max = fmap_max)
        # 创建 StyleVectorizer、Generator 和 Discriminator 实例

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul = lr_mul)
        self.GE = Generator(image_size, latent_dim, network_capacity, transparent = transparent, no_const = no_const)
        # 创建额外的 StyleVectorizer 和 Generator 实例

        self.D_aug = AugWrapper(self.D, image_size, aug_types)
        # 创建用于增强所有输入到鉴别器的包装器

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)
        # 设置 SE 和 GE 的梯度计算为 False

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = Adam(generator_params, lr = self.lr, betas=(0.5, 0.9))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr * ttur_mult, betas=(0.5, 0.9))
        # 设置生成器和鉴别器的优化器

        self._init_weights()
        self.reset_parameter_averaging()
        # 初始化权重和参数平均化

        self.cuda()
        # 将模型移至 GPU

        self.fp16 = fp16
        if fp16:
            (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1')
        # 如果启用混合精度训练，则初始化混合精度训练

    def _init_weights(self):
        # 初始化权重函数
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        # 对卷积层和全连接层进行权重初始化

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)
        # 初始化生成器中的噪声层参数

    def EMA(self):
        # 指数移动平均函数
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)
        # 更新移动平均参数

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)
        # 更新 SE 和 GE 的移动平均参数

    def reset_parameter_averaging(self):
        # 重置参数平均化函数
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())
        # 将 SE 和 GE 的状态字典加载到 S 和 G 中

    def forward(self, x):
        # 前向传播函数
        return x
        # 返回输入 x

class Trainer():
    # 定义 Trainer 类
    # 初始化函数，设置模型参数和训练参数
    def __init__(self, name, results_dir, models_dir, image_size, network_capacity, transparent = False, batch_size = 4, mixed_prob = 0.9, gradient_accumulate_every=1, lr = 2e-4, ttur_mult = 2, num_workers = None, save_every = 1000, trunc_psi = 0.6, fp16 = False, no_const = False, aug_prob = 0., dataset_aug_prob = 0., cr_weight = 0.2, apply_pl_reg = False, lr_mul = 0.1, *args, **kwargs):
        # 存储 GAN 参数
        self.GAN_params = [args, kwargs]
        self.GAN = None

        # 设置模型名称、结果目录、模型目录、配置文件路径
        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'

        # 检查图像大小是否为2的幂次方
        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.transparent = transparent

        self.no_const = no_const
        self.aug_prob = aug_prob

        # 设置学习率、TTUR倍数、学习率倍数、批量大小、工作进程数、混合概率
        self.lr = lr
        self.ttur_mult = ttur_mult
        self.lr_mul = lr_mul
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.apply_pl_reg = apply_pl_reg
        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        # 检查是否支持混合精度训练
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex is not available for you to use mixed precision training'
        self.fp16 = fp16

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0
        self.last_cr_loss = 0

        # 初始化指数移动平均
        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        self.cr_weight = cr_weight

    # 初始化 GAN 模型
    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(lr = self.lr, ttur_mult = self.ttur_mult, lr_mul = self.lr_mul, image_size = self.image_size, network_capacity = self.network_capacity, transparent = self.transparent, fp16 = self.fp16, no_const = self.no_const, *args, **kwargs)

    # 写入配置文件
    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    # 加载配置文件
    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        self.transparent = config['transparent']
        self.no_const = config.pop('no_const', False)
        del self.GAN
        self.init_GAN()

    # 返回配置信息
    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity, 'transparent': self.transparent, 'no_const': self.no_const}

    # 设置数据源
    def set_data_src(self, folder):
        self.dataset = Dataset(folder, self.image_size, transparent = self.transparent, aug_prob = self.dataset_aug_prob)
        self.loader = cycle(data.DataLoader(self.dataset, num_workers = default(self.num_workers, num_cores), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))

    # 禁用梯度计算
    @torch.no_grad()
    # 定义评估函数，用于生成图像
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0):
        # 将 GAN 设置为评估模式
        self.GAN.eval()
        # 根据是否透明设置文件扩展名
        ext = 'jpg' if not self.transparent else 'png'
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        # 生成潜在向量和噪声
        latents = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # regular

        # 生成正常图像
        generated_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        
        # moving averages

        # 生成移动平均图像
        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

        # mixing regularities

        # 定义瓷砖函数
        def tile(a, dim, n_tile):
            init_dim = a.size(dim)
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*(repeat_idx))
            order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
            return torch.index_select(a, dim, order_index)

        nn = noise(num_rows, latent_dim)
        tmp1 = tile(nn, 0, num_rows)
        tmp2 = nn.repeat(num_rows, 1)

        tt = int(num_layers / 2)
        mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        # 生成混合图像
        generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    # 生成截断图像
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8):
        latent_dim = G.latent_dim

        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)
            
        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_torch = torch.from_numpy(self.av).cuda()
            tmp = trunc_psi * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    @torch.no_grad()
    # 生成插值图像序列
    def generate_interpolation(self, num = 0, num_image_tiles = 8, trunc = 1.0, save_frames = False):
        # 将 GAN 设置为评估模式
        self.GAN.eval()
        # 确定文件扩展名
        ext = 'jpg' if not self.transparent else 'png'
        # 设置图像行数
        num_rows = num_image_tiles

        # 获取潜在空间维度、图像尺寸和层数
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # 生成潜在向量和噪声
        latents_low = noise(num_rows ** 2, latent_dim)
        latents_high = noise(num_rows ** 2, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # 创建插值比例
        ratios = torch.linspace(0., 8., 100)

        frames = []
        # 遍历插值比例
        for ratio in tqdm(ratios):
            # 线性插值生成插值潜在向量
            interp_latents = slerp(ratio, latents_low, latents_high)
            latents = [(interp_latents, num_layers)]
            # 生成经过截断的图像
            generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, latents, n, trunc_psi = self.trunc_psi)
            # 将生成的图像拼接成网格
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            # 转换为 PIL 图像
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            frames.append(pil_image)

        # 保存为 GIF 动画
        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        # 如果需要保存每一帧图像
        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}')

    # 打印日志信息
    def print_log(self):
        pl_mean = default(self.pl_mean, 0)
        print(f'G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {pl_mean:.2f} | CR: {self.last_cr_loss:.2f}')

    # 返回模型文件名
    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    # 初始化结果和模型文件夹
    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    # 清空结果和模型文件夹
    def clear(self):
        rmtree(f'./models/{self.name}', True)
        rmtree(f'./results/{self.name}', True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    # 保存模型
    def save(self, num):
        save_data = {'GAN': self.GAN.state_dict()}

        if self.GAN.fp16:
            save_data['amp'] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    # 加载模型
    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        self.GAN.load_state_dict(load_data['GAN'])

        if self.GAN.fp16 and 'amp' in load_data:
            amp.load_state_dict(load_data['amp'])
```