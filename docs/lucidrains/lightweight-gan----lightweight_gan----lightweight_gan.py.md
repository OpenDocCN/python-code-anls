# `.\lucidrains\lightweight-gan\lightweight_gan\lightweight_gan.py`

```
# 导入必要的库
import os
import json
import multiprocessing
from random import random
import math
from math import log2, floor
from functools import lru_cache, partial
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
from kornia.filters import filter2d

from lightweight_gan.diff_augment import DiffAugment
from lightweight_gan.version import __version__

from tqdm import tqdm
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from adabelief_pytorch import AdaBelief

# 断言，检查是否有可用的 CUDA 加速
assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

# 常量定义
NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png', 'tiff']

# 辅助函数

# 检查值是否存在
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

# 检查值是否为2的幂
def is_power_of_two(val):
    return log2(val).is_integer()

# 返回默认值
def default(val, d):
    return val if exists(val) else d

# 设置模型参数是否需要梯度
def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool

# 无限循环生成器
def cycle(iterable):
    while True:
        for i in iterable:
            yield i

# 如果值为 NaN，则抛出异常
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

# 将输入数据按照最大批次大小分块处理
def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

# 球面插值函数
def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

# 安全除法函数
def safe_div(n, d):
    try:
        res = n / d
    except ZeroDivisionError:
        prefix = '' if int(n >= 0) else '-'
        res = float(f'{prefix}inf')
    return res

# 损失函数

# 生成器 Hinge Loss
def gen_hinge_loss(fake, real):
    return fake.mean()

# Hinge Loss
def hinge_loss(real, fake):
    return (F.relu(1 + real) + F.relu(1 - fake)).mean()

# 双对比损失函数
def dual_contrastive_loss(real_logits, fake_logits):
    device = real_logits.device
    real_logits, fake_logits = map(lambda t: rearrange(t, '... -> (...)'), (real_logits, fake_logits))

    def loss_half(t1, t2):
        t1 = rearrange(t1, 'i -> i ()')
        t2 = repeat(t2, 'j -> i j', i = t1.shape[0])
        t = torch.cat((t1, t2), dim = -1)
        return F.cross_entropy(t, torch.zeros(t1.shape[0], device = device, dtype = torch.long))

    return loss_half(real_logits, fake_logits) + loss_half(-fake_logits, -real_logits)

# 缓存随机数生成器
@lru_cache(maxsize=10)
def det_randn(*args):
    """
    deterministic random to track the same latent vars (and images) across training steps
    # 用于在训练步骤中可视化相同图像
    """
    # 返回一个具有指定形状的随机张量
    return torch.randn(*args)
# 定义一个函数，用于在两个向量之间插值生成多个样本
def interpolate_between(a, b, *, num_samples, dim):
    # 断言样本数量大于2
    assert num_samples > 2
    # 初始化样本列表
    samples = []
    # 初始化步长
    step_size = 0
    # 循环生成插值样本
    for _ in range(num_samples):
        # 使用线性插值生成样本
        sample = torch.lerp(a, b, step_size)
        samples.append(sample)
        # 更新步长
        step_size += 1 / (num_samples - 1)
    # 将生成的样本堆叠在一起
    return torch.stack(samples, dim=dim)

# 辅助类

# 定义一个自定义异常类
class NanException(Exception):
    pass

# 定义一个指数移动平均类
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
    def update_average(self, old, new):
        # 如果旧值不存在，则直接返回新值
        if not exists(old):
            return new
        # 计算新的指数移动平均值
        return old * self.beta + (1 - self.beta) * new

# 定义一个随机应用类
class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else = lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob
    def forward(self, x):
        # 根据概率选择应用哪个函数
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)

# 定义一个通道归一化类
class ChanNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # 计算均值和方差
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        # 执行通道归一化操作
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

# 定义一个预归一化类
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = ChanNorm(dim)

    def forward(self, x):
        # 执行预归一化操作
        return self.fn(self.norm(x))

# 定义一个残差连接类
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        # 执行残差连接操作
        return self.fn(x) + x

# 定义一个分支求和类
class SumBranches(nn.Module):
    def __init__(self, branches):
        super().__init__()
        self.branches = nn.ModuleList(branches)
    def forward(self, x):
        # 对分支函数的输出进行求和
        return sum(map(lambda fn: fn(x), self.branches))

# 定义一个模糊类
class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)
    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f [None, :, None]
        return filter2d(x, f, normalized=True)

# 定义一个噪声类
class Noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise = None):
        b, _, h, w, device = *x.shape, x.device

        # 如果噪声不存在，则生成随机噪声
        if not exists(noise):
            noise = torch.randn(b, 1, h, w, device = device)

        return x + self.weight * noise

# 定义一个二维卷积函数，保持输入输出尺寸相同
def Conv2dSame(dim_in, dim_out, kernel_size, bias = True):
    pad_left = kernel_size // 2
    pad_right = (pad_left - 1) if (kernel_size % 2) == 0 else pad_left

    return nn.Sequential(
        nn.ZeroPad2d((pad_left, pad_right, pad_left, pad_right)),
        nn.Conv2d(dim_in, dim_out, kernel_size, bias = bias)
    )

# 注意力机制

# 定义一个深度卷积类
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LinearAttention(nn.Module):
    # 初始化函数，设置注意力头数、头维度、卷积核大小等参数
    def __init__(self, dim, dim_head = 64, heads = 8, kernel_size = 3):
        # 调用父类初始化函数
        super().__init__()
        # 计算缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        self.kernel_size = kernel_size
        # 使用 GELU 作为非线性激活函数
        self.nonlin = nn.GELU()

        # 线性变换，将输入特征映射到内部维度
        self.to_lin_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_lin_kv = DepthWiseConv2d(dim, inner_dim * 2, 3, padding = 1, bias = False)

        # 线性变换，将输入特征映射到内部维度
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias = False)

        # 输出层线性变换，将内部维度映射回原始维度
        self.to_out = nn.Conv2d(inner_dim * 2, dim, 1)

    # 前向传播函数
    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        # 线性注意力计算

        lin_q, lin_k, lin_v = (self.to_lin_q(fmap), *self.to_lin_kv(fmap).chunk(2, dim = 1))
        lin_q, lin_k, lin_v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (lin_q, lin_k, lin_v))

        lin_q = lin_q.softmax(dim = -1)
        lin_k = lin_k.softmax(dim = -2)

        lin_q = lin_q * self.scale

        context = einsum('b n d, b n e -> b d e', lin_k, lin_v)
        lin_out = einsum('b n d, b d e -> b n e', lin_q, context)
        lin_out = rearrange(lin_out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # 类卷积的全局注意力计算

        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) c x y', h = h), (q, k, v))

        k = F.unfold(k, kernel_size = self.kernel_size, padding = self.kernel_size // 2)
        v = F.unfold(v, kernel_size = self.kernel_size, padding = self.kernel_size // 2)

        k, v = map(lambda t: rearrange(t, 'b (d j) n -> b n j d', d = self.dim_head), (k, v))

        q = rearrange(q, 'b c ... -> b (...) c') * self.scale

        sim = einsum('b i d, b i j d -> b i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()

        attn = sim.softmax(dim = -1)

        full_out = einsum('b i j, b i j d -> b i d', attn, v)
        full_out = rearrange(full_out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        # 将线性注意力和类卷积全局注意力的输出相加

        lin_out = self.nonlin(lin_out)
        out = torch.cat((lin_out, full_out), dim = 1)
        return self.to_out(out)
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
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        # 如果通道数与目标通道数相同，则返回原始张量
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

        # 如果不存在 alpha 通道且需要透明度，则创建全为1的 alpha 通道
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

        # 根据是否需要透明度和是否为灰度图像确定通道数和 Pillow 模式
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

        # 图像转换操作���列
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

# 数据增强包装类
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

# 规范化类
norm_class = nn.BatchNorm2d

# 像素混洗上采样类
class PixelShuffleUpsample(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
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

    # 前向传播函数，将输入数据传递给网络并返回输出
    def forward(self, x):
        return self.net(x)
def SPConvDownsample(dim, dim_out = None):
    # 定义一个下采样函数，根据输入维度和输出维度进行下采样
    # 在论文 https://arxiv.org/abs/2208.03641 中显示这是最优的下采样方式
    # 在论文中被称为 SP-conv，实际上是像素解缩放
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )

# squeeze excitation classes

# 全局上下文网络
# https://arxiv.org/abs/2012.13375
# 类似于 squeeze-excite，但具有简化的注意力池化和随后的层归一化

class GlobalContext(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out
    ):
        super().__init__()
        self.to_k = nn.Conv2d(chan_in, 1, 1)
        chan_intermediate = max(3, chan_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        context = self.to_k(x)
        context = context.flatten(2).softmax(dim = -1)
        out = einsum('b i n, b c n -> b c i', context, x.flatten(2))
        out = out.unsqueeze(-1)
        return self.net(out)

# 频道注意力

# 获取一维离散余弦变换
def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    return result * (1 if freq == 0 else math.sqrt(2))

# 获取离散余弦变换权重
def get_dct_weights(width, channel, fidx_u, fidx_v):
    dct_weights = torch.zeros(1, channel, width, width)
    c_part = channel // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for x in range(width):
            for y in range(width):
                coor_value = get_1d_dct(x, u_x, width) * get_1d_dct(y, v_y, width)
                dct_weights[:, i * c_part: (i + 1) * c_part, x, y] = coor_value

    return dct_weights

class FCANet(nn.Module):
    def __init__(
        self,
        *,
        chan_in,
        chan_out,
        reduction = 4,
        width
    ):
        super().__init__()

        freq_w, freq_h = ([0] * 8), list(range(8)) # 在论文中，似乎16个频率是理想的
        dct_weights = get_dct_weights(width, chan_in, [*freq_w, *freq_h], [*freq_h, *freq_w])
        self.register_buffer('dct_weights', dct_weights)

        chan_intermediate = max(3, chan_out // reduction)

        self.net = nn.Sequential(
            nn.Conv2d(chan_in, chan_intermediate, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(chan_intermediate, chan_out, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = reduce(x * self.dct_weights, 'b c (h h1) (w w1) -> b c h1 w1', 'sum', h1 = 1, w1 = 1)
        return self.net(x)

# 生成对抗网络

class Generator(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        latent_dim = 256,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        attn_res_layers = [],
        freq_chan_attn = False
    ):
        # 调用父类的构造函数
        super().__init__()
        # 计算图像分辨率的对数值
        resolution = log2(image_size)
        # 断言图像大小必须是2的幂次方
        assert is_power_of_two(image_size), 'image size must be a power of 2'

        # 根据是否透明或灰度图像确定初始通道数
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        # 设置特征图的最大通道数
        fmap_max = default(fmap_max, latent_dim)

        # 初始化卷积层
        self.initial_conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim * 2, 4),
            norm_class(latent_dim * 2),
            nn.GLU(dim = 1)
        )

        # 计算层数和特征
        num_layers = int(resolution) - 2
        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), range(2, num_layers + 2)))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))
        features = list(map(lambda n: 3 if n[0] >= 8 else n[1], features))
        features = [latent_dim, *features]

        # 计算输入输出特征
        in_out_features = list(zip(features[:-1], features[1:]))

        # 初始化残差层和特征映射
        self.res_layers = range(2, num_layers + 2)
        self.layers = nn.ModuleList([])
        self.res_to_feature_map = dict(zip(self.res_layers, in_out_features))

        # 设置空间尺寸映射
        self.sle_map = ((3, 7), (4, 8), (5, 9), (6, 10))
        self.sle_map = list(filter(lambda t: t[0] <= resolution and t[1] <= resolution, self.sle_map))
        self.sle_map = dict(self.sle_map)

        self.num_layers_spatial_res = 1

        # 遍历每一层并构建网络层
        for (res, (chan_in, chan_out)) in zip(self.res_layers, in_out_features):
            image_width = 2 ** res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            sle = None
            if res in self.sle_map:
                residual_layer = self.sle_map[res]
                sle_chan_out = self.res_to_feature_map[residual_layer - 1][-1]

                if freq_chan_attn:
                    sle = FCANet(
                        chan_in = chan_out,
                        chan_out = sle_chan_out,
                        width = 2 ** (res + 1)
                    )
                else:
                    sle = GlobalContext(
                        chan_in = chan_out,
                        chan_out = sle_chan_out
                    )

            layer = nn.ModuleList([
                nn.Sequential(
                    PixelShuffleUpsample(chan_in),
                    Blur(),
                    Conv2dSame(chan_in, chan_out * 2, 4),
                    Noise(),
                    norm_class(chan_out * 2),
                    nn.GLU(dim = 1)
                ),
                sle,
                attn
            ])
            self.layers.append(layer)

        # 输出卷积层
        self.out_conv = nn.Conv2d(features[-1], init_channel, 3, padding = 1)

    # 前向传播函数
    def forward(self, x):
        # 重排输入张量的维度
        x = rearrange(x, 'b c -> b c () ()')
        x = self.initial_conv(x)
        # 对输入张量进行归一化
        x = F.normalize(x, dim = 1)

        residuals = dict()

        # 遍历每一层并执行前向传播
        for (res, (up, sle, attn)) in zip(self.res_layers, self.layers):
            if exists(attn):
                x = attn(x) + x

            x = up(x)

            if exists(sle):
                out_res = self.sle_map[res]
                residual = sle(x)
                residuals[out_res] = residual

            next_res = res + 1
            if next_res in residuals:
                x = x * residuals[next_res]

        return self.out_conv(x)
# 定义一个简单的解码器类，继承自 nn.Module
class SimpleDecoder(nn.Module):
    # 初始化函数，接受输入通道数、输出通道数、上采样次数等参数
    def __init__(
        self,
        *,
        chan_in,
        chan_out = 3,
        num_upsamples = 4,
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化层列表
        self.layers = nn.ModuleList([])
        # 设置最终输出通道数
        final_chan = chan_out
        # 设置初始通道数
        chans = chan_in

        # 循环创建上采样层
        for ind in range(num_upsamples):
            # 判断是否为最后一层
            last_layer = ind == (num_upsamples - 1)
            # 根据是否为最后一层确定输出通道数
            chan_out = chans if not last_layer else final_chan * 2
            # 创建包含上采样、卷积和 GLU 激活函数的层
            layer = nn.Sequential(
                PixelShuffleUpsample(chans),
                nn.Conv2d(chans, chan_out, 3, padding = 1),
                nn.GLU(dim = 1)
            )
            # 将层添加到层列表中
            self.layers.append(layer)
            # 更新通道数
            chans //= 2

    # 前向传播函数
    def forward(self, x):
        # 遍历所有层并依次进行前向传播
        for layer in self.layers:
            x = layer(x)
        # 返回输出结果
        return x

# 定义一个鉴别器类，继承自 nn.Module
class Discriminator(nn.Module):
    # 初始化函数，接受输入图像大小、最大特征图数、特征图反比系数、是否透明、是否灰度、输出尺寸、注意力机制层等参数
    def __init__(
        self,
        *,
        image_size,
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = []
        ):
        # 调用父类的构造函数
        super().__init__()
        # 计算图像分辨率的对数值
        resolution = log2(image_size)
        # 断言图像大小必须是2的幂次方
        assert is_power_of_two(image_size), 'image size must be a power of 2'
        # 断言判别器输出维度只能是5x5或1x1
        assert disc_output_size in {1, 5}, 'discriminator output dimensions can only be 5x5 or 1x1'

        resolution = int(resolution)

        # 根据是否透明或灰度图像确定初始通道数
        if transparent:
            init_channel = 4
        elif greyscale:
            init_channel = 1
        else:
            init_channel = 3

        # 计算非残差层的数量
        num_non_residual_layers = max(0, int(resolution) - 8)
        num_residual_layers = 8 - 3

        # 计算非残差层的分辨率范围
        non_residual_resolutions = range(min(8, resolution), 2, -1)
        # 计算特征通道数
        features = list(map(lambda n: (n,  2 ** (fmap_inverse_coef - n)), non_residual_resolutions))
        features = list(map(lambda n: (n[0], min(n[1], fmap_max)), features))

        # 如果没有非残差层，则将初始通道数赋给第一个特征通道数
        if num_non_residual_layers == 0:
            res, _ = features[0]
            features[0] = (res, init_channel)

        # 将特征通道数组合成输入输出通道数的列表
        chan_in_out = list(zip(features[:-1], features[1:]))

        # 初始化非残差层
        self.non_residual_layers = nn.ModuleList([])
        for ind in range(num_non_residual_layers):
            first_layer = ind == 0
            last_layer = ind == (num_non_residual_layers - 1)
            chan_out = features[0][-1] if last_layer else init_channel

            self.non_residual_layers.append(nn.Sequential(
                Blur(),
                nn.Conv2d(init_channel, chan_out, 4, stride = 2, padding = 1),
                nn.LeakyReLU(0.1)
            ))

        # 初始化残差层
        self.residual_layers = nn.ModuleList([])

        for (res, ((_, chan_in), (_, chan_out))) in zip(non_residual_resolutions, chan_in_out):
            image_width = 2 ** res

            attn = None
            if image_width in attn_res_layers:
                attn = PreNorm(chan_in, LinearAttention(chan_in))

            self.residual_layers.append(nn.ModuleList([
                SumBranches([
                    nn.Sequential(
                        Blur(),
                        SPConvDownsample(chan_in, chan_out),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(chan_out, chan_out, 3, padding = 1),
                        nn.LeakyReLU(0.1)
                    ),
                    nn.Sequential(
                        Blur(),
                        nn.AvgPool2d(2),
                        nn.Conv2d(chan_in, chan_out, 1),
                        nn.LeakyReLU(0.1),
                    )
                ]),
                attn
            ]))

        # 获取最后一个特征通道数
        last_chan = features[-1][-1]
        # 根据判别器输出大小选择不同的输出层结构
        if disc_output_size == 5:
            self.to_logits = nn.Sequential(
                nn.Conv2d(last_chan, last_chan, 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )
        elif disc_output_size == 1:
            self.to_logits = nn.Sequential(
                Blur(),
                nn.Conv2d(last_chan, last_chan, 3, stride = 2, padding = 1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(last_chan, 1, 4)
            )

        # 初始化形状判别器输出层
        self.to_shape_disc_out = nn.Sequential(
            nn.Conv2d(init_channel, 64, 3, padding = 1),
            Residual(PreNorm(64, LinearAttention(64))),
            SumBranches([
                nn.Sequential(
                    Blur(),
                    SPConvDownsample(64, 32),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(32, 32, 3, padding = 1),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    Blur(),
                    nn.AvgPool2d(2),
                    nn.Conv2d(64, 32, 1),
                    nn.LeakyReLU(0.1),
                )
            ]),
            Residual(PreNorm(32, LinearAttention(32))),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Conv2d(32, 1, 4)
        )

        # 初始化解码器1
        self.decoder1 = SimpleDecoder(chan_in = last_chan, chan_out = init_channel)
        # 如果分辨率大于等于9，则初始化解码器2
        self.decoder2 = SimpleDecoder(chan_in = features[-2][-1], chan_out = init_channel) if resolution >= 9 else None
    # 前向传播函数，接受输入 x 和是否计算辅助损失 calc_aux_loss
    def forward(self, x, calc_aux_loss = False):
        # 保存原始输入图像
        orig_img = x

        # 遍历非残差层并计算输出
        for layer in self.non_residual_layers:
            x = layer(x)

        # 初始化存储每个残差块输出的列表
        layer_outputs = []

        # 遍历残差层，计算输出并存储在列表中
        for (net, attn) in self.residual_layers:
            # 如果存在注意力机制，将注意力机制应用到输入上并与输入相加
            if exists(attn):
                x = attn(x) + x

            # 经过残差块网络
            x = net(x)
            # 将输出添加到列表中
            layer_outputs.append(x)

        # 将最终输出转换为 logits 并展平
        out = self.to_logits(x).flatten(1)

        # 将原始图像插值为 32x32 大小
        img_32x32 = F.interpolate(orig_img, size = (32, 32))
        # 将插值后的图像传入形状判别器
        out_32x32 = self.to_shape_disc_out(img_32x32)

        # 如果不需要计算辅助损失，则直接返回结果
        if not calc_aux_loss:
            return out, out_32x32, None

        # 自监督自编码损失

        # 获取倒数第一个残差块的输出
        layer_8x8 = layer_outputs[-1]
        # 获取倒数第二个残差块的输出
        layer_16x16 = layer_outputs[-2]

        # 使用解码器1对 8x8 层进行重建
        recon_img_8x8 = self.decoder1(layer_8x8)

        # 计算 MSE 损失
        aux_loss = F.mse_loss(
            recon_img_8x8,
            F.interpolate(orig_img, size = recon_img_8x8.shape[2:])
        )

        # 如果存在第二个解码器
        if exists(self.decoder2):
            # 随机选择一个象限
            select_random_quadrant = lambda rand_quadrant, img: rearrange(img, 'b c (m h) (n w) -> (m n) b c h w', m = 2, n = 2)[rand_quadrant]
            crop_image_fn = partial(select_random_quadrant, floor(random() * 4))
            img_part, layer_16x16_part = map(crop_image_fn, (orig_img, layer_16x16))

            # 使用解码器2对 16x16 层进行重建
            recon_img_16x16 = self.decoder2(layer_16x16_part)

            # 计算 MSE 损失
            aux_loss_16x16 = F.mse_loss(
                recon_img_16x16,
                F.interpolate(img_part, size = recon_img_16x16.shape[2:])
            )

            # 将两个损失相加
            aux_loss = aux_loss + aux_loss_16x16

        # 返回最终结果，包括主要输出、32x32 输出和辅助损失
        return out, out_32x32, aux_loss
# 定义 LightweightGAN 类，继承自 nn.Module
class LightweightGAN(nn.Module):
    # 初始化函数，接收多个参数
    def __init__(
        self,
        *,
        latent_dim,
        image_size,
        optimizer = "adam",
        fmap_max = 512,
        fmap_inverse_coef = 12,
        transparent = False,
        greyscale = False,
        disc_output_size = 5,
        attn_res_layers = [],
        freq_chan_attn = False,
        ttur_mult = 1.,
        lr = 2e-4,
        rank = 0,
        ddp = False
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化类的属性
        self.latent_dim = latent_dim
        self.image_size = image_size

        # 定义 G_kwargs 字典
        G_kwargs = dict(
            image_size = image_size,
            latent_dim = latent_dim,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            freq_chan_attn = freq_chan_attn
        )

        # 创建 Generator 对象
        self.G = Generator(**G_kwargs)

        # 创建 Discriminator 对象
        self.D = Discriminator(
            image_size = image_size,
            fmap_max = fmap_max,
            fmap_inverse_coef = fmap_inverse_coef,
            transparent = transparent,
            greyscale = greyscale,
            attn_res_layers = attn_res_layers,
            disc_output_size = disc_output_size
        )

        # 创建 EMA 对象
        self.ema_updater = EMA(0.995)
        # 创建 Generator 对象 GE
        self.GE = Generator(**G_kwargs)
        # 设置 GE 不需要梯度
        set_requires_grad(self.GE, False)

        # 根据 optimizer 参数选择优化器
        if optimizer == "adam":
            self.G_opt = Adam(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = Adam(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        elif optimizer == "adabelief":
            self.G_opt = AdaBelief(self.G.parameters(), lr = lr, betas=(0.5, 0.9))
            self.D_opt = AdaBelief(self.D.parameters(), lr = lr * ttur_mult, betas=(0.5, 0.9))
        else:
            assert False, "No valid optimizer is given"

        # 初始化权重
        self.apply(self._init_weights)
        # 重置参数平均
        self.reset_parameter_averaging()

        # 将模型移动到 GPU
        self.cuda(rank)
        # 创建 D_aug 对象
        self.D_aug = AugWrapper(self.D, image_size)

    # 初始化权重函数
    def _init_weights(self, m):
        if type(m) in {nn.Conv2d, nn.Linear}:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    # 更新指数移动平均函数
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

    # 前向传播函数，抛出异常
    def forward(self, x):
        raise NotImplemented

# trainer

class Trainer():
    # 初始化函数，设置各种参数的默认值
    def __init__(
        self,
        name = 'default',  # 模型名称，默认为'default'
        results_dir = 'results',  # 结果保存目录，默认为'results'
        models_dir = 'models',  # 模型保存目录，默认为'models'
        base_dir = './',  # 基础目录，默认为当前目录
        optimizer = 'adam',  # 优化器，默认为'adam'
        num_workers = None,  # 工作进程数，默认为None
        latent_dim = 256,  # 潜在空间维度，默认为256
        image_size = 128,  # 图像尺寸，默认为128
        num_image_tiles = 8,  # 图像平铺数，默认为8
        fmap_max = 512,  # 特征图最大数量，默认为512
        transparent = False,  # 是否透明，默认为False
        greyscale = False,  # 是否灰度，默认为False
        batch_size = 4,  # 批量大小，默认为4
        gp_weight = 10,  # 梯度惩罚权重，默认为10
        gradient_accumulate_every = 1,  # 梯度积累频率，默认为1
        attn_res_layers = [],  # 注意力机制层，默认为空列表
        freq_chan_attn = False,  # 频道注意力，默认为False
        disc_output_size = 5,  # 判别器输出大小，默认为5
        dual_contrast_loss = False,  # 双对比损失，默认为False
        antialias = False,  # 抗锯齿，默认为False
        lr = 2e-4,  # 学习率，默认为2e-4
        lr_mlp = 1.,  # 学习率倍增，默认为1.0
        ttur_mult = 1.,  # TTUR倍增，默认为1.0
        save_every = 1000,  # 每隔多少步保存模型，默认为1000
        evaluate_every = 1000,  # 每隔多少步评估模型，默认为1000
        aug_prob = None,  # 数据增强概率，默认为None
        aug_types = ['translation', 'cutout'],  # 数据增强类型，默认为['translation', 'cutout']
        dataset_aug_prob = 0.,  # 数据集增强概率，默认为0.0
        calculate_fid_every = None,  # 计算FID频率，默认为None
        calculate_fid_num_images = 12800,  # 计算FID所需图像数量，默认为12800
        clear_fid_cache = False,  # 清除FID缓存，默认为False
        is_ddp = False,  # 是否使用分布式数据并行，默认为False
        rank = 0,  # 进程排名，默认为0
        world_size = 1,  # 进程总数，默认为1
        log = False,  # 是否记录日志，默认为False
        amp = False,  # 是否使用自动混合精度，默认为False
        hparams = None,  # 超参数，默认为None
        use_aim = True,  # 是否使用AIM，默认为True
        aim_repo = None,  # AIM仓库，默认为None
        aim_run_hash = None,  # AIM运行哈希，默认为None
        load_strict = True,  # 是否严格加��模型，默认为True
        *args,  # 可变位置参数
        **kwargs  # 可变关键字参数
        ):
        # 初始化 GAN 参数为传入的参数和关键字参数的元组
        self.GAN_params = [args, kwargs]
        # 初始化 GAN 为 None
        self.GAN = None

        # 设置名称
        self.name = name

        # 将 base_dir 转换为 Path 对象
        base_dir = Path(base_dir)
        self.base_dir = base_dir
        # 设置结果目录和模型目录
        self.results_dir = base_dir / results_dir
        self.models_dir = base_dir / models_dir
        self.fid_dir = base_dir / 'fid' / name

        # 设置配置文件路径
        self.config_path = self.models_dir / name / '.config.json'

        # 检查图像大小是否为 2 的幂次方
        assert is_power_of_two(image_size), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        # 检查注意力分辨率层是否都为 2 的幂次方
        assert all(map(is_power_of_two, attn_res_layers)), 'resolution layers of attention must all be powers of 2 (16, 32, 64, 128, 256, 512)'

        # 检查是否使用双对比损失时鉴别器输出大小是否大于 1
        assert not (dual_contrast_loss and disc_output_size > 1), 'discriminator output size cannot be greater than 1 if using dual contrastive loss'

        # 设置图像大小和图像瓦片数量
        self.image_size = image_size
        self.num_image_tiles = num_image_tiles

        # 设置潜在空间维度、特征图最大值、透明度和灰度
        self.latent_dim = latent_dim
        self.fmap_max = fmap_max
        self.transparent = transparent
        self.greyscale = greyscale

        # 检查是否只设置了透明度或灰度
        assert (int(self.transparent) + int(self.greyscale)) < 2, 'you can only set either transparency or greyscale'

        # 设置数据增强概率和类型
        self.aug_prob = aug_prob
        self.aug_types = aug_types

        # 设置学习率、优化器、工作进程数、TTUR 倍数、批量大小、梯度累积步数
        self.lr = lr
        self.optimizer = optimizer
        self.num_workers = num_workers
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        # 设置梯度惩罚权重
        self.gp_weight = gp_weight

        # 设置评估和保存频率
        self.evaluate_every = evaluate_every
        self.save_every = save_every
        self.steps = 0

        # 设置注意力分辨率层和频道注意力
        self.attn_res_layers = attn_res_layers
        self.freq_chan_attn = freq_chan_attn

        # 设置鉴别���输出大小和抗锯齿
        self.disc_output_size = disc_output_size
        self.antialias = antialias

        # 设置双对比损失
        self.dual_contrast_loss = dual_contrast_loss

        # 初始化损失和 FID
        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = None
        self.last_recon_loss = None
        self.last_fid = None

        # 初始化文件夹
        self.init_folders()

        # 初始化数据加载器和数据集增强概率
        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob

        # 设置计算 FID 的频率和图像数量
        self.calculate_fid_every = calculate_fid_every
        self.calculate_fid_num_images = calculate_fid_num_images
        self.clear_fid_cache = clear_fid_cache

        # 设置是否使用分布式数据并行
        self.is_ddp = is_ddp
        self.is_main = rank == 0
        self.rank = rank
        self.world_size = world_size

        # 设置是否使用同步批归一化
        self.syncbatchnorm = is_ddp

        # 设置加载严格性
        self.load_strict = load_strict

        # 设置混合精度训练和梯度缩放器
        self.amp = amp
        self.G_scaler = GradScaler(enabled = self.amp)
        self.D_scaler = GradScaler(enabled = self.amp)

        # 初始化运行和超参数
        self.run = None
        self.hparams = hparams

        # 如果是主进程且使用 AIM
        if self.is_main and use_aim:
            try:
                import aim
                self.aim = aim
            except ImportError:
                print('unable to import aim experiment tracker - please run `pip install aim` first')

            # 创建 AIM 实验追踪器
            self.run = self.aim.Run(run_hash=aim_run_hash, repo=aim_repo)
            self.run['hparams'] = hparams

    # 图像扩展名属性
    @property
    def image_extension(self):
        return 'jpg' if not self.transparent else 'png'

    # 检查点编号属性
    @property
    def checkpoint_num(self):
        return floor(self.steps // self.save_every)
    # 初始化 GAN 模型
    def init_GAN(self):
        # 获取 GAN 参数
        args, kwargs = self.GAN_params

        # 在实例化 GAN 之前设置一些全局变量

        global norm_class
        global Blur

        # 根据条件选择使用 SyncBatchNorm 还是 BatchNorm2d
        norm_class = nn.SyncBatchNorm if self.syncbatchnorm else nn.BatchNorm2d
        # 根据条件选择使用 Identity 还是 Blur
        Blur = nn.Identity if not self.antialias else Blur

        # 处理从多 GPU 切换回单 GPU 时的 bug

        if self.syncbatchnorm and not self.is_ddp:
            import torch.distributed as dist
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group('nccl', rank=0, world_size=1)

        # 实例化 GAN

        self.GAN = LightweightGAN(
            optimizer=self.optimizer,
            lr = self.lr,
            latent_dim = self.latent_dim,
            attn_res_layers = self.attn_res_layers,
            freq_chan_attn = self.freq_chan_attn,
            image_size = self.image_size,
            ttur_mult = self.ttur_mult,
            fmap_max = self.fmap_max,
            disc_output_size = self.disc_output_size,
            transparent = self.transparent,
            greyscale = self.greyscale,
            rank = self.rank,
            *args,
            **kwargs
        )

        if self.is_ddp:
            ddp_kwargs = {'device_ids': [self.rank], 'output_device': self.rank, 'find_unused_parameters': True}

            # 使用分布式数据并行处理模型
            self.G_ddp = DDP(self.GAN.G, **ddp_kwargs)
            self.D_ddp = DDP(self.GAN.D, **ddp_kwargs)
            self.D_aug_ddp = DDP(self.GAN.D_aug, **ddp_kwargs)

    # 写入配置信息
    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    # 加载配置信息
    def load_config(self):
        # 如果配置文件不存在，则使用默认配置
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        # 更新配置信息
        self.image_size = config['image_size']
        self.transparent = config['transparent']
        self.syncbatchnorm = config['syncbatchnorm']
        self.disc_output_size = config['disc_output_size']
        self.greyscale = config.pop('greyscale', False)
        self.attn_res_layers = config.pop('attn_res_layers', [])
        self.freq_chan_attn = config.pop('freq_chan_attn', False)
        self.optimizer = config.pop('optimizer', 'adam')
        self.fmap_max = config.pop('fmap_max', 512)
        del self.GAN
        # 重新初始化 GAN 模型
        self.init_GAN()

    # 返回配置信息
    def config(self):
        return {
            'image_size': self.image_size,
            'transparent': self.transparent,
            'greyscale': self.greyscale,
            'syncbatchnorm': self.syncbatchnorm,
            'disc_output_size': self.disc_output_size,
            'optimizer': self.optimizer,
            'attn_res_layers': self.attn_res_layers,
            'freq_chan_attn': self.freq_chan_attn
        }

    # 设置数据源
    def set_data_src(self, folder):
        # 计算并设置数据加载器的工作进程数
        num_workers = default(self.num_workers, math.ceil(NUM_CORES / self.world_size))
        # 创建数据集
        self.dataset = ImageDataset(folder, self.image_size, transparent = self.transparent, greyscale = self.greyscale, aug_prob = self.dataset_aug_prob)
        # 创建分布式采样器
        sampler = DistributedSampler(self.dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True) if self.is_ddp else None
        # 创建数据加载器
        dataloader = DataLoader(self.dataset, num_workers = num_workers, batch_size = math.ceil(self.batch_size / self.world_size), sampler = sampler, shuffle = not self.is_ddp, drop_last = True, pin_memory = True)
        self.loader = cycle(dataloader)

        # 如果数据集检测到样本数量较少，则自动设置数据增强概率
        num_samples = len(self.dataset)
        if not exists(self.aug_prob) and num_samples < 1e5:
            self.aug_prob = min(0.5, (1e5 - num_samples) * 3e-6)
            print(f'autosetting augmentation probability to {round(self.aug_prob * 100)}%')

    # 禁用梯度计算
    @torch.no_grad()
    # 定义一个评估函数，用于生成图像
    def evaluate(self, num = 0, num_image_tiles = 4):
        # 将 GAN 设置为评估模式
        self.GAN.eval()

        # 获取图像文件的扩展名
        ext = self.image_extension
        # 设置图像展示的行数
        num_rows = num_image_tiles
    
        # 获取潜在空间的维度和图像的尺寸
        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # 将图像转换为 PIL 格式的函数
        def image_to_pil(image):
            # 将图像转换为 PIL 图像格式
            ndarr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            return im

        # 生成潜在空间和噪声
        latents = det_randn((num_rows ** 2, latent_dim)).cuda(self.rank)
        interpolate_latents = interpolate_between(latents[:num_rows], latents[-num_rows:],
                                                  num_samples=num_rows,
                                                  dim=0).flatten(end_dim=1)

        # 生成插值图像
        generate_interpolations = self.generate_(self.GAN.G, interpolate_latents)
        if self.run is not None:
            # 将生成的插值图像分组
            grouped = generate_interpolations.view(num_rows, num_rows, *generate_interpolations.shape[1:])
            for idx, images in enumerate(grouped):
                alpha = idx / (len(grouped) - 1)
                aim_images = []
                for image in images:
                    im = image_to_pil(image)
                    aim_images.append(self.aim.Image(im, caption=f'#{idx}'))

                # 跟踪生成的图像
                self.run.track(value=aim_images, name='generated',
                               step=self.steps,
                               context={'interpolated': True,
                                        'alpha': alpha})
        # 保存生成的插值图像
        torchvision.utils.save_image(generate_interpolations, str(self.results_dir / self.name / f'{str(num)}-interp.{ext}'), nrow=num_rows)
        
        # 生成正常图像
        generated_images = self.generate_(self.GAN.G, latents)

        if self.run is not None:
            aim_images = []
            for idx, image in enumerate(generated_images):
                im = image_to_pil(image)
                aim_images.append(self.aim.Image(im, caption=f'#{idx}'))

            # 跟踪生成的图像
            self.run.track(value=aim_images, name='generated',
                           step=self.steps,
                           context={'ema': False})
        # 保存生成的正常图像
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

        # 生成移动平均图像
        generated_images = self.generate_(self.GAN.GE, latents)
        if self.run is not None:
            aim_images = []
            for idx, image in enumerate(generated_images):
                im = image_to_pil(image)
                aim_images.append(self.aim.Image(im, caption=f'EMA #{idx}'))

            # 跟踪生成的图像
            self.run.track(value=aim_images, name='generated',
                           step=self.steps,
                           context={'ema': True})
        # 保存生成的移动平均图像
        torchvision.utils.save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-ema.{ext}'), nrow=num_rows)

    # 禁用梯度计算
    @torch.no_grad()
    # 生成图片，可以指定生成数量、图像瓦片数量、检查点、类型
    def generate(self, num=0, num_image_tiles=4, checkpoint=None, types=['default', 'ema']):
        # 将 GAN 设置为评估模式
        self.GAN.eval()

        # 获取潜在空间维度
        latent_dim = self.GAN.latent_dim
        # 生成目录名
        dir_name = self.name + str('-generated-') + str(checkpoint)
        # 生成完整目录路径
        dir_full = Path().absolute() / self.results_dir / dir_name
        # 图像文件扩展名
        ext = self.image_extension

        # 如果目录不存在，则创建
        if not dir_full.exists():
            os.mkdir(dir_full)

        # 生成默认类型的图片
        if 'default' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated default images'):
                # 生成随机潜在向量
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                # 生成图片
                generated_image = self.generate_(self.GAN.G, latents)
                # 生成图片路径
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}.{ext}')
                # 保存生成的图片
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        # 生成EMA类型的图片
        if 'ema' in types:
            for i in tqdm(range(num_image_tiles), desc='Saving generated EMA images'):
                # 生成随机潜在向量
                latents = torch.randn((1, latent_dim)).cuda(self.rank)
                # 生成图片
                generated_image = self.generate_(self.GAN.GE, latents)
                # 生成图片路径
                path = str(self.results_dir / dir_name / f'{str(num)}-{str(i)}-ema.{ext}')
                # 保存生成的图片
                torchvision.utils.save_image(generated_image[0], path, nrow=1)

        # 返回生成图片的目录路径
        return dir_full

    # 用于显示训练进度的方法
    @torch.no_grad()
    def show_progress(self, num_images=4, types=['default', 'ema']):
        # 获取所有检查点
        checkpoints = self.get_checkpoints()
        # 检查是否存在检查点
        assert exists(checkpoints), 'cannot find any checkpoints to create a training progress video for'

        # 进度目录名
        dir_name = self.name + str('-progress')
        # 进度完整目录路径
        dir_full = Path().absolute() / self.results_dir / dir_name
        # 图像文件扩展名
        ext = self.image_extension
        # 潜在向量初始��为None
        latents = None

        # 计算检查点数字的位数
        zfill_length = math.ceil(math.log10(len(checkpoints)))

        # 如果目录不存在，则创建
        if not dir_full.exists():
            os.mkdir(dir_full)

        # 遍历所有检查点
        for checkpoint in tqdm(checkpoints, desc='Generating progress images'):
            # 加载模型参数
            self.load(checkpoint, print_version=False)
            # 将 GAN 设置为评估模式
            self.GAN.eval()

            # 如果是第一个检查点，生成随机潜在向量
            if checkpoint == 0:
                latents = torch.randn((num_images, self.GAN.latent_dim)).cuda(self.rank)

            # 生成默认类型的图片
            if 'default' in types:
                generated_image = self.generate_(self.GAN.G, latents)
                # 生成图片路径
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}.{ext}')
                # 保存生成的图片
                torchvision.utils.save_image(generated_image, path, nrow=num_images)

            # 生成EMA��型的图片
            if 'ema' in types:
                generated_image = self.generate_(self.GAN.GE, latents)
                # 生成图片路径
                path = str(self.results_dir / dir_name / f'{str(checkpoint).zfill(zfill_length)}-ema.{ext}')
                # 保存生成的图片
                torchvision.utils.save_image(generated_image, path, nrow=num_images)

    # 用于禁用梯度计算的装饰器
    @torch.no_grad()
    # 计算 FID 分数
    def calculate_fid(self, num_batches):
        # 导入 FID 分数计算模块
        from pytorch_fid import fid_score
        # 清空 GPU 缓存
        torch.cuda.empty_cache()

        # 设置真实图片和生成图片的路径
        real_path = self.fid_dir / 'real'
        fake_path = self.fid_dir / 'fake'

        # 如果不存在真实图片路径或需要清除 FID 缓存，则删除现有文件并重新创建目录
        if not real_path.exists() or self.clear_fid_cache:
            rmtree(real_path, ignore_errors=True)
            os.makedirs(real_path)

            # 保存真实图片
            for batch_num in tqdm(range(num_batches), desc='calculating FID - saving reals'):
                real_batch = next(self.loader)
                for k, image in enumerate(real_batch.unbind(0)):
                    ind = k + batch_num * self.batch_size
                    torchvision.utils.save_image(image, real_path / f'{ind}.png')

        # 删除生成图片路径并重新创建目录
        rmtree(fake_path, ignore_errors=True)
        os.makedirs(fake_path)

        # 设置生成器为评估模式
        self.GAN.eval()
        ext = self.image_extension

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # 生成假图片
        for batch_num in tqdm(range(num_batches), desc='calculating FID - saving generated'):
            # 生成潜在向量和噪声
            latents = torch.randn(self.batch_size, latent_dim).cuda(self.rank)

            # 生成图片
            generated_images = self.generate_(self.GAN.GE, latents)

            for j, image in enumerate(generated_images.unbind(0)):
                ind = j + batch_num * self.batch_size
                torchvision.utils.save_image(image, str(fake_path / f'{str(ind)}-ema.{ext}'))

        # 返回 FID 分数
        return fid_score.calculate_fid_given_paths([str(real_path), str(fake_path)], 256, latents.device, 2048)

    # 生成图片
    @torch.no_grad()
    def generate_(self, G, style, num_image_tiles = 8):
        # 评估生成器
        generated_images = evaluate_in_chunks(self.batch_size, G, style)
        return generated_images.clamp_(0., 1.)

    # 生成插值图片
    @torch.no_grad()
    def generate_interpolation(self, num = 0, num_image_tiles = 8, num_steps = 100, save_frames = False):
        # 设置生成器为评估模式
        self.GAN.eval()
        ext = self.image_extension
        num_rows = num_image_tiles

        latent_dim = self.GAN.latent_dim
        image_size = self.GAN.image_size

        # 生成潜在向量和噪声
        latents_low = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)
        latents_high = torch.randn(num_rows ** 2, latent_dim).cuda(self.rank)

        # 生成插值比例
        ratios = torch.linspace(0., 8., num_steps)

        frames = []
        for ratio in tqdm(ratios):
            # 线性插值生成潜在向量
            interp_latents = slerp(ratio, latents_low, latents_high)
            generated_images = self.generate_(self.GAN.GE, interp_latents)
            images_grid = torchvision.utils.make_grid(generated_images, nrow = num_rows)
            pil_image = transforms.ToPILImage()(images_grid.cpu())
            
            # 如果需要透明背景，则设置透明度
            if self.transparent:
                background = Image.new('RGBA', pil_image.size, (255, 255, 255))
                pil_image = Image.alpha_composite(background, pil_image)
                
            frames.append(pil_image)

        # 保存插值图片为 GIF
        frames[0].save(str(self.results_dir / self.name / f'{str(num)}.gif'), save_all=True, append_images=frames[1:], duration=80, loop=0, optimize=True)

        # 如果需要保存每一帧图片
        if save_frames:
            folder_path = (self.results_dir / self.name / f'{str(num)}')
            folder_path.mkdir(parents=True, exist_ok=True)
            for ind, frame in enumerate(frames):
                frame.save(str(folder_path / f'{str(ind)}.{ext}'))
    # 打印训练日志信息
    def print_log(self):
        # 定义包含损失信息的数据列表
        data = [
            ('G', self.g_loss),
            ('D', self.d_loss),
            ('GP', self.last_gp_loss),
            ('SS', self.last_recon_loss),
            ('FID', self.last_fid)
        ]

        # 过滤掉值为 None 的数据
        data = [d for d in data if exists(d[1])]
        # 将数据转换为字符串格式，用 '|' 连接
        log = ' | '.join(map(lambda n: f'{n[0]}: {n[1]:.2f}', data))
        # 打印日志信息
        print(log)

        # 如果存在运行实例，则追踪数据
        if self.run is not None:
            for key, value in data:
                self.run.track(value, key, step=self.steps)

        # 返回数据列表
        return data

    # 返回模型文件名
    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    # 初始化文件夹
    def init_folders(self):
        # 创建结果目录和模型目录
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    # 清空文件夹
    def clear(self):
        # 删除模型目录、结果目录、FID 目录和配置文件路径
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.fid_dir), True)
        rmtree(str(self.config_path), True)
        # 初始化文件夹
        self.init_folders()

    # 保存模型
    def save(self, num):
        # 保存模型相关数据
        save_data = {
            'GAN': self.GAN.state_dict(),
            'version': __version__,
            'G_scaler': self.G_scaler.state_dict(),
            'D_scaler': self.D_scaler.state_dict()
        }

        # 将数据保存到模型文件中
        torch.save(save_data, self.model_name(num))
        # 写入配置文件
        self.write_config()

    # 加载模型
    def load(self, num=-1, print_version=True):
        # 加载配置文件
        self.load_config()

        name = num
        if num == -1:
            # 获取已保存的检查点
            checkpoints = self.get_checkpoints()

            if not exists(checkpoints):
                return

            name = checkpoints[-1]
            print(f'continuing from previous epoch - {name}')

        self.steps = name * self.save_every

        # 加载模型数据
        load_data = torch.load(self.model_name(name))

        if print_version and 'version' in load_data and self.is_main:
            print(f"loading from version {load_data['version']}")

        try:
            self.GAN.load_state_dict(load_data['GAN'], strict=self.load_strict)
        except Exception as e:
            saved_version = load_data['version']
            print('unable to load save model. please try downgrading the package to the version specified by the saved model (to do so, just run `pip install lightweight-gan=={saved_version}`')
            raise e

        if 'G_scaler' in load_data:
            self.G_scaler.load_state_dict(load_data['G_scaler'])
        if 'D_scaler' in load_data:
            self.D_scaler.load_state_dict(load_data['D_scaler'])

    # 获取已保存的检查点
    def get_checkpoints(self):
        # 获取模型目录下所有模型文件路径
        file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
        # 提取已保存的模型编号
        saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))

        if len(saved_nums) == 0:
            return None

        return saved_nums
```