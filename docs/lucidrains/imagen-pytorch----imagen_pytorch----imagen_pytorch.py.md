# `.\lucidrains\imagen-pytorch\imagen_pytorch\imagen_pytorch.py`

```py
# 导入数学库
import math
# 从随机模块中导入随机函数
from random import random
# 从 beartype 库中导入 List 和 Union 类型
from beartype.typing import List, Union
# 从 beartype 库中导入 beartype 装饰器
from beartype import beartype
# 从 tqdm 库中导入 tqdm 函数
from tqdm.auto import tqdm
# 从 functools 库中导入 partial 和 wraps 函数
from functools import partial, wraps
# 从 contextlib 库中导入 contextmanager 和 nullcontext 函数
from contextlib import contextmanager, nullcontext
# 从 pathlib 库中导入 Path 类

from pathlib import Path

# 导入 torch 库
import torch
# 从 torch.nn.functional 模块中导入 F 函数
import torch.nn.functional as F
# 从 torch.nn.parallel 模块中导入 DistributedDataParallel 类
from torch.nn.parallel import DistributedDataParallel
# 从 torch 模块中导入 nn 和 einsum 函数
from torch import nn, einsum
# 从 torch.cuda.amp 模块中导入 autocast 函数
from torch.cuda.amp import autocast
# 从 torch.special 模块中导入 expm1 函数
from torch.special import expm1
# 从 torchvision.transforms 模块中导入 T 函数

import torchvision.transforms as T

# 从 kornia.augmentation 模块中导入 K 函数
import kornia.augmentation as K

# 从 einops 模块中导入 rearrange, repeat, reduce, pack, unpack 函数
from einops import rearrange, repeat, reduce, pack, unpack
# 从 einops.layers.torch 模块中导入 Rearrange 类
from einops.layers.torch import Rearrange

# 从 imagen_pytorch.t5 模块中导入 t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME 函数
from imagen_pytorch.t5 import t5_encode_text, get_encoded_dim, DEFAULT_T5_NAME

# 从 imagen_pytorch.imagen_video 模块中导入 Unet3D, resize_video_to, scale_video_time 函数

from imagen_pytorch.imagen_video import Unet3D, resize_video_to, scale_video_time

# helper functions

# 判断值是否存在
def exists(val):
    return val is not None

# 返回输入值
def identity(t, *args, **kwargs):
    return t

# 判断一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 返回列表的第一个元素，如果列表为空则返回默认值
def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

# 可能的装饰器
def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

# 仅执行一次的装饰器
def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

# 仅打印一次的装饰器
print_once = once(print)

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 将输入值转换为元组
def cast_tuple(val, length = None):
    if isinstance(val, list):
        val = tuple(val)

    output = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length):
        assert len(output) == length

    return output

# 压缩字典，去除值为 None 的键值对
def compact(input_dict):
    return {key: value for key, value in input_dict.items() if exists(value)}

# 对字典中指定键的值进行转换
def maybe_transform_dict_key(input_dict, key, fn):
    if key not in input_dict:
        return input_dict

    copied_dict = input_dict.copy()
    copied_dict[key] = fn(copied_dict[key])
    return copied_dict

# 将 uint8 类型的图像转换为 float 类型
def cast_uint8_images_to_float(images):
    if not images.dtype == torch.uint8:
        return images
    return images / 255

# 获取模块的设备信息
def module_device(module):
    return next(module.parameters()).device

# 初始化权重为零
def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

# 模型评估装饰器
def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# 将元组填充到指定长度
def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))

# helper classes

# 空操作模块
class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

# tensor helpers

# 计算张量的对数
def log(t, eps: float = 1e-12):
    return torch.log(t.clamp(min = eps))

# 计算张量的 L2 范数
def l2norm(t):
    return F.normalize(t, dim = -1)

# 将一个张量的维度右侧填充到与另一个张量相同的维度
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# 计算带有掩码的张量均值
def masked_mean(t, *, dim, mask = None):
    if not exists(mask):
        return t.mean(dim = dim)

    denom = mask.sum(dim = dim, keepdim = True)
    mask = rearrange(mask, 'b n -> b n 1')
    masked_t = t.masked_fill(~mask, 0.)

    return masked_t.sum(dim = dim) / denom.clamp(min = 1e-5)

# 调整图像大小
def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    mode = 'nearest'
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    out = F.interpolate(image, target_image_size, mode = mode)

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

# 计算所有帧的维度
def calc_all_frame_dims(
    downsample_factors: List[int],
    frames
):
    # 如果frames不存在，则返回一个空元组的元组，长度为downsample_factors的长度
    if not exists(frames):
        return (tuple(),) * len(downsample_factors)

    # 存储所有帧的维度信息
    all_frame_dims = []

    # 遍历downsample_factors列表
    for divisor in downsample_factors:
        # 断言frames能够被divisor整除
        assert divisible_by(frames, divisor)
        # 将frames除以divisor得到的结果作为元组添加到all_frame_dims列表中
        all_frame_dims.append((frames // divisor,))

    # 返回所有帧的维度信息
    return all_frame_dims
# 安全获取元组中指定索引的值，如果索引超出范围则返回默认值
def safe_get_tuple_index(tup, index, default = None):
    if len(tup) <= index:
        return default
    return tup[index]

# 图像归一化函数
# ddpms 期望图像范围在 -1 到 1 之间

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

# 无分类器指导函数

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# 连续时间高斯扩散辅助函数和类
# 这部分很大程度上要感谢 @crowsonkb 在 https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

@torch.jit.script
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # 不确定这是否考虑了在离散版本中 beta 被剪切为 0.999

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class GaussianDiffusionContinuousTimes(nn.Module):
    def __init__(self, *, noise_schedule, timesteps = 1000):
        super().__init__()

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.float32)

    def sample_random_times(self, batch_size, *, device):
        return torch.zeros((batch_size,), device = device).float().uniform_(0, 1)

    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.num_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        t_next = default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise = None):
        dtype = x_start.dtype

        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device = x_start.device, dtype = dtype)

        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t).type(dtype)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        return alpha * x_start + sigma * noise, log_snr, alpha, sigma
    # 从输入的 x_from 中采样数据，从 from_t 到 to_t 时间范围内，添加噪声
    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        # 获取输入 x_from 的形状、设备和数据类型
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]

        # 如果 from_t 是浮点数，则将其转换为与 batch 大小相同的张量
        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device = device, dtype = dtype)

        # 如果 to_t 是浮点数，则将其转换为与 batch 大小相同的张量
        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device = device, dtype = dtype)

        # 如果未提供噪声，则生成一个与 x_from 相同形状的随机噪声张量
        noise = default(noise, lambda: torch.randn_like(x_from))

        # 计算 from_t 对应的 log_snr，并将其维度与 x_from 对齐
        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        # 根据 log_snr 计算 alpha 和 sigma
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        # 计算 to_t 对应的 log_snr，并将其维度与 x_from 对齐
        log_snr_to = self.log_snr(to_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        # 根据 log_snr_to 计算 alpha_to 和 sigma_to
        alpha_to, sigma_to =  log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        # 返回根据公式计算得到的结果
        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    # 根据给定的 x_t、t 和速度 v 预测起始值
    def predict_start_from_v(self, x_t, t, v):
        # 计算 t 对应的 log_snr，并将其维度与 x_t 对齐
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        # 根据 log_snr 计算 alpha 和 sigma
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        # 返回根据公式计算得到的结果
        return alpha * x_t - sigma * v

    # 根据给定的 x_t、t 和噪声 noise 预测起始值
    def predict_start_from_noise(self, x_t, t, noise):
        # 计算 t 对应的 log_snr，并将其维度与 x_t 对齐
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        # 根据 log_snr 计算 alpha 和 sigma
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        # 返回根据公式计算得到的结果
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)
# 定义 LayerNorm 类，用于实现层归一化操作
class LayerNorm(nn.Module):
    # 初始化函数，接受特征数、是否稳定、维度作为参数
    def __init__(self, feats, stable = False, dim = -1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        # 初始化可学习参数 g
        self.g = nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    # 前向传播函数
    def forward(self, x):
        dtype, dim = x.dtype, self.dim

        # 如果设置了稳定性，对输入进行归一化处理
        if self.stable:
            x = x / x.amax(dim = dim, keepdim = True).detach()

        # 根据数据类型选择 eps 值
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        # 计算方差和均值
        var = torch.var(x, dim = dim, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = dim, keepdim = True)

        # 返回归一化后的结果
        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)

# 定义 ChanLayerNorm 类，是 LayerNorm 的一个特例，维度为 -3
ChanLayerNorm = partial(LayerNorm, dim = -3)

# 定义 Always 类，用于返回固定值
class Always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val

# 定义 Residual 类，实现残差连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# 定义 Parallel 类，实现并行计算
class Parallel(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)

# 定义 PerceiverAttention 类，实现注意力机制
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        # 初始化层归一化操作和线性变换
        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        # 初始化缩放参数
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 输出层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.LayerNorm(dim)
        )

    # 前向传播函数
    def forward(self, x, latents, mask = None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # 拼接键值对
        kv_input = torch.cat((x, latents), dim = -2)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # 对 q 和 k 进行 L2 归一化
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # 计算相似度并进行掩码处理
        sim = einsum('... i d, ... j d  -> ... i j', q, k) * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # 注意力计算
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

# 定义 PerceiverResampler 类，实现 Perceiver 模型的重采样
class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_latents_mean_pooled = 4, # number of latents derived from mean pooled representation of the sequence
        max_seq_len = 512,
        ff_mult = 4
    # 初始化函数，继承父类的初始化方法
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 创建位置编码的嵌入层，用于将位置信息嵌入输入数据中
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # 创建可学习的潜在变量，用于表示输入数据的潜在特征
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # 初始化从平均池化序列到潜在变量的映射层
        self.to_latents_from_mean_pooled_seq = None

        # 如果平均池化的潜在变量数量大于0，则创建映射层
        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled)
            )

        # 创建多层感知器的注意力和前馈网络层
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

    # 前向传播函数，接收输入数据 x 和掩码 mask
    def forward(self, x, mask = None):
        # 获取输入数据的长度和设备信息
        n, device = x.shape[1], x.device
        # 根据位置编码获取位置嵌入
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        # 将输入数据与位置编码相加，融合位置信息
        x_with_pos = x + pos_emb

        # 重复潜在变量以匹配输入数据的维度
        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        # 如果存在平均池化的潜在变量映射层，则将平均池化的潜在变量与原始潜在变量拼接
        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(x, dim = 1, mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim = -2)

        # 遍历多层感知器的注意力和前馈网络层
        for attn, ff in self.layers:
            # 使用注意力层处理输入数据和潜在变量，然后与潜在变量相加
            latents = attn(x_with_pos, latents, mask = mask) + latents
            # 使用前馈网络层处理潜在变量，然后与潜在变量相加
            latents = ff(latents) + latents

        # 返回处理后的潜在变量
        return latents
# 定义注意力机制模块
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        context_dim = None,
        scale = 8
    ):
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context = None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = map(lambda t: repeat(t, 'd -> b 1 d', b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # add text conditioning, if present

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            k = torch.cat((ck, k), dim = -2)
            v = torch.cat((cv, v), dim = -2)

        # qk rmsnorm

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义上采样函数
def Upsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

# 定义像素混洗上采样类
class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
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

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

# 定义下采样函数
def Downsample(dim, dim_out = None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    # 返回一个包含两个操作的序列：1. 重新排列输入张量的维度，将其转换为'b (c s1 s2) h w'的形式；2. 使用1x1卷积层将输入通道数从dim * 4降至dim_out
    return nn.Sequential(
        # 重新排列输入张量的维度，将其转换为'b (c s1 s2) h w'的形式，其中s1和s2分别为2
        Rearrange('b c (h s1) (w s2) -> b (c s1 s2) h w', s1 = 2, s2 = 2),
        # 使用1x1卷积层将输入通道数从dim * 4降至dim_out
        nn.Conv2d(dim * 4, dim_out, 1)
    )
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)  # 计算对数值
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)  # 计算指数值
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')  # 重排张量形状
        return torch.cat((emb.sin(), emb.cos()), dim = -1)  # 拼接正弦和余弦值

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))  # 初始化权重参数

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')  # 重排张量形状
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi  # 计算频率
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)  # 拼接正弦和余弦值
        fouriered = torch.cat((x, fouriered), dim = -1)  # 拼接原始张量和傅立叶变换结果
        return fouriered

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8,
        norm = True
    ):
        super().__init__()
        self.groupnorm = nn.GroupNorm(groups, dim) if norm else Identity()  # 初始化分组归一化层
        self.activation = nn.SiLU()  # 激活函数
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)  # 卷积层

    def forward(self, x, scale_shift = None):
        x = self.groupnorm(x)  # 分组归一化

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # 缩放和平移

        x = self.activation(x)  # 激活函数
        return self.project(x)  # 卷积操作

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8,
        linear_attn = False,
        use_gca = False,
        squeeze_excite = False,
        **attn_kwargs
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )  # 时间条件的多层感��机

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = attn_klass(
                dim = dim_out,
                context_dim = cond_dim,
                **attn_kwargs
            )  # 交叉注意力机制

        self.block1 = Block(dim, dim_out, groups = groups)  # 第一个块
        self.block2 = Block(dim_out, dim_out, groups = groups)  # 第二个块

        self.gca = GlobalContext(dim_in = dim_out, dim_out = dim_out) if use_gca else Always(1)  # 全局上下文注意力

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else Identity()  # 残差卷积

    def forward(self, x, time_emb = None, cond = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)  # 分割时间嵌入

        h = self.block1(x)  # 第一个块操作

        if exists(self.cross_attn):
            assert exists(cond)
            h = rearrange(h, 'b c h w -> b h w c')
            h, ps = pack([h], 'b * c')
            h = self.cross_attn(h, context = cond) + h  # 交叉注意力机制
            h, = unpack(h, ps, 'b * c')
            h = rearrange(h, 'b h w c -> b c h w')

        h = self.block2(h, scale_shift = scale_shift)  # 第二个块操作

        h = h * self.gca(h)  # 全局上下文注意力

        return h + self.res_conv(x)  # 返回残差连接结果

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        norm_context = False,
        scale = 8
    # 初始化函数，设置缩放因子和头数
    def __init__(
        super().__init__()
        self.scale = scale

        self.heads = heads
        inner_dim = dim_head * heads

        # 设置上下文维度
        context_dim = default(context_dim, dim)

        # 初始化层归一化
        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

        # 初始化空键值对
        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        # 线性变换，将输入转换为查询向量
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 线性变换，将上下文转换为键值对
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        # 初始化查询和键的缩放参数
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 输出层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    # 前向传播函数
    def forward(self, x, context, mask = None):
        # 获取输入的形状和设备信息
        b, n, device = *x.shape[:2], x.device

        # 对输入和上下文进行层归一化
        x = self.norm(x)
        context = self.norm_context(context)

        # 获取查询、键、值
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        # 重排查询、键、值的维度
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # 添加空键/值对，用于分类器在先验网络中的自由引导
        nk, nv = map(lambda t: repeat(t, 'd -> b h 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # 余弦相似度注意力
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        # 计算相似度
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # 掩码
        max_neg_value = -torch.finfo(sim.dtype).max
        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # softmax计算注意力权重
        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.to(sim.dtype)

        # 加权求和得到输出
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
class LinearCrossAttention(CrossAttention):
    # 线性交叉注意力类，继承自CrossAttention类
    def forward(self, x, context, mask = None):
        # 前向传播函数，接收输入x、上下文context和掩码mask，默认为None
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        # 对输入x进行规范化处理
        context = self.norm_context(context)
        # 对上下文context进行规范化处理

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        # 将输入x和上下文context转换为查询q、键k和值v

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = self.heads), (q, k, v))
        # 对查询q、键k和值v进行形状重排

        # add null key / value for classifier free guidance in prior net
        # 在先前网络中添加空键/值以用于分类器的自由引导

        nk, nv = map(lambda t: repeat(t, 'd -> (b h) 1 d', h = self.heads,  b = b), self.null_kv.unbind(dim = -2))

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # masking
        # 掩码处理

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b n -> b n 1')
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.)

        # linear attention
        # 线性注意力计算

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = self.heads)
        return self.to_out(out)

class LinearAttention(nn.Module):
    # 线性注意力类，继承自nn.Module类
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8,
        dropout = 0.05,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias = False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias = False, padding = 1, groups = inner_dim)
        )

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, inner_dim * 2, bias = False)) if exists(context_dim) else None

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap, context = None):
        # 前向传播函数，接收特征图fmap和上下文context，默认为None
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = h), (q, k, v))

        if exists(context):
            assert exists(self.to_context)
            ck, cv = self.to_context(context).chunk(2, dim = -1)
            ck, cv = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (ck, cv))
            k = torch.cat((k, ck), dim = -2)
            v = torch.cat((v, cv), dim = -2)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class GlobalContext(nn.Module):
    # 全局上下文类
    """ basically a superior form of squeeze-excitation that is attention-esque """

    def __init__(
        self,
        *,
        dim_in,
        dim_out
    # 定义一个类，继承自 nn.Module
    class Attention(nn.Module):
        # 初始化函数
        def __init__(self, dim_in, dim_out):
            # 调用父类的初始化函数
            super().__init__()
            # 创建一个卷积层，输入维度为 dim_in，输出维度为 1，卷积核大小为 1
            self.to_k = nn.Conv2d(dim_in, 1, 1)
            # 计算隐藏层维度，取 dim_out 除以 2 和 3 中的较大值
            hidden_dim = max(3, dim_out // 2)
    
            # 创建一个神经网络序列
            self.net = nn.Sequential(
                # 第一层卷积层，输入维度为 dim_in，输出维度为 hidden_dim，卷积核大小为 1
                nn.Conv2d(dim_in, hidden_dim, 1),
                # 使用 SiLU 激活函数
                nn.SiLU(),
                # 第二层卷积层，输入维度为 hidden_dim，输出维度为 dim_out，卷积核大小为 1
                nn.Conv2d(hidden_dim, dim_out, 1),
                # 使用 Sigmoid 激活函数
                nn.Sigmoid()
            )
    
        # 前向传播函数
        def forward(self, x):
            # 将输入 x 通过 self.to_k 进行处理，得到 context
            context = self.to_k(x)
            # 对 x 和 context 进行维度重排，将 'b n ...' 转换为 'b n (...)'
            x, context = map(lambda t: rearrange(t, 'b n ... -> b n (...)'), (x, context))
            # 使用 einsum 进行张量乘法，计算注意力权重
            out = einsum('b i n, b c n -> b c i', context.softmax(dim = -1), x)
            # 将输出 out 进行维度重排，将 '...' 转换为 '... 1'
            out = rearrange(out, '... -> ... 1')
            # 将处理后的 out 输入到神经网络 self.net 中
            return self.net(out)
# 定义一个前馈神经网络模块，包含层归一化、线性层、GELU激活函数和线性层
def FeedForward(dim, mult = 2):
    # 计算隐藏层维度
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),  # 层归一化
        nn.Linear(dim, hidden_dim, bias = False),  # 线性层
        nn.GELU(),  # GELU激活函数
        LayerNorm(hidden_dim),  # 层归一化
        nn.Linear(hidden_dim, dim, bias = False)  # 线性层
    )

# 定义一个通道前馈神经网络模块，包含通道层归一化、卷积层、GELU激活函数和卷积层
def ChanFeedForward(dim, mult = 2):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),  # 通道层归一化
        nn.Conv2d(dim, hidden_dim, 1, bias = False),  # 卷积层
        nn.GELU(),  # GELU激活函数
        ChanLayerNorm(hidden_dim),  # 通道层归一化
        nn.Conv2d(hidden_dim, dim, 1, bias = False)  # 卷积层
    )

# 定义一个Transformer块，包含多个自注意力层和前馈神经网络层
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),  # 自注意力层
                FeedForward(dim = dim, mult = ff_mult)  # 前馈神经网络层
            ]))

    def forward(self, x, context = None):
        x = rearrange(x, 'b c h w -> b h w c')
        x, ps = pack([x], 'b * c')

        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x

        x, = unpack(x, ps, 'b * c')
        x = rearrange(x, 'b h w c -> b c h w')
        return x

# 定义一个线性注意力Transformer块，包含多个线性注意力层和通道前馈神经网络层
class LinearAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth = 1,
        heads = 8,
        dim_head = 32,
        ff_mult = 2,
        context_dim = None,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LinearAttention(dim = dim, heads = heads, dim_head = dim_head, context_dim = context_dim),  # 线性注意力层
                ChanFeedForward(dim = dim, mult = ff_mult)  # 通道前馈神经网络层
            ]))

    def forward(self, x, context = None):
        for attn, ff in self.layers:
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x

# 定义一个交叉嵌入层，包含多个卷积层
class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # 计算每个尺度的维度
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

# 定义一个上采样合并器，包含多个块
class UpsampleCombiner(nn.Module):
    def __init__(
        self,
        dim,
        *,
        enabled = False,
        dim_ins = tuple(),
        dim_outs = tuple()
    ):
        super().__init__()
        dim_outs = cast_tuple(dim_outs, len(dim_ins))
        assert len(dim_ins) == len(dim_outs)

        self.enabled = enabled

        if not self.enabled:
            self.dim_out = dim
            return

        self.fmap_convs = nn.ModuleList([Block(dim_in, dim_out) for dim_in, dim_out in zip(dim_ins, dim_outs)])
        self.dim_out = dim + (sum(dim_outs) if len(dim_outs) > 0 else 0)
    # 定义一个前向传播函数，接受输入 x 和特征图列表 fmaps，默认为 None
    def forward(self, x, fmaps = None):
        # 获取输入 x 的最后一个维度大小作为目标大小
        target_size = x.shape[-1]

        # 如果未提供特征图列表，则使用空元组
        fmaps = default(fmaps, tuple())

        # 如果模块未启用，特征图列表为空，或者卷积层列表为空，则直接返回输入 x
        if not self.enabled or len(fmaps) == 0 or len(self.fmap_convs) == 0:
            return x

        # 将特征图列表中的每个特征图调整大小为目标大小
        fmaps = [resize_image_to(fmap, target_size) for fmap in fmaps]
        # 对每个调整大小后的特征图应用对应的卷积操作，得到输出列表
        outs = [conv(fmap) for fmap, conv in zip(fmaps, self.fmap_convs)]
        # 在第一个维度上拼接输入 x 和所有输出，返回结果
        return torch.cat((x, *outs), dim = 1)
# 定义一个名为 Unet 的类，继承自 nn.Module
class Unet(nn.Module):
    # 初始化方法，设置类的属性
    def __init__(
        self,
        *,
        dim,
        text_embed_dim = get_encoded_dim(DEFAULT_T5_NAME),  # 默认文本嵌入维度
        num_resnet_blocks = 1,  # ResNet 块的数量
        cond_dim = None,  # 条件维度
        num_image_tokens = 4,  # 图像令牌数量
        num_time_tokens = 2,  # 时间令牌数量
        learned_sinu_pos_emb_dim = 16,  # 学习的正弦位置编码维度
        out_dim = None,  # 输出维度
        dim_mults=(1, 2, 4, 8),  # 维度倍增
        cond_images_channels = 0,  # 条件图像通道数
        channels = 3,  # 通道数
        channels_out = None,  # 输出通道数
        attn_dim_head = 64,  # 注意力头维度
        attn_heads = 8,  # 注意力头数量
        ff_mult = 2.,  # FeedForward 层倍增因子
        lowres_cond = False,  # 低分辨率条件
        layer_attns = True,  # 层间注意力
        layer_attns_depth = 1,  # 层间注意力深度
        layer_mid_attns_depth = 1,  # 中间层注意力深度
        layer_attns_add_text_cond = True,  # 是否使用文本嵌入来条件化自注意力块
        attend_at_middle = True,  # 是否在瓶颈处进行注意力
        layer_cross_attns = True,  # 层间交叉注意力
        use_linear_attn = False,  # 是否使用线性注意力
        use_linear_cross_attn = False,  # 是否使用线性交叉注意力
        cond_on_text = True,  # 是否在文本上进行条件化
        max_text_len = 256,  # 最大文本长度
        init_dim = None,  # 初始化维度
        resnet_groups = 8,  # ResNet 组数
        init_conv_kernel_size = 7,  # 初始卷积核大小
        init_cross_embed = True,  # 初始化交叉嵌入
        init_cross_embed_kernel_sizes = (3, 7, 15),  # 初始化交叉嵌入的卷积核大小
        cross_embed_downsample = False,  # 交叉嵌入下采样
        cross_embed_downsample_kernel_sizes = (2, 4),  # 交叉嵌入下采样的卷积核大小
        attn_pool_text = True,  # 注意力池化文本
        attn_pool_num_latents = 32,  # 注意力池化潜在数
        dropout = 0.,  # 丢弃率
        memory_efficient = False,  # 内存效率
        init_conv_to_final_conv_residual = False,  # 初始卷积到最终卷积的残差连接
        use_global_context_attn = True,  # 使用全局上下文注意力
        scale_skip_connection = True,  # 缩放跳跃连接
        final_resnet_block = True,  # 最终 ResNet 块
        final_conv_kernel_size = 3,  # 最终卷积核大小
        self_cond = False,  # 自条件
        resize_mode = 'nearest',  # 调整模式
        combine_upsample_fmaps = False,  # 合并所有上采样块的特征图
        pixel_shuffle_upsample = True,  # 像素混洗上采样
    # 如果当前 Unet 的设置不正确，重新使用正确的设置重新初始化 Unet
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        text_embed_dim,
        channels,
        channels_out,
        cond_on_text
    ):
        # 如果设置与当前 Unet 的设置相同，则返回当前 Unet
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_text == self.cond_on_text and \
            text_embed_dim == self._locals['text_embed_dim'] and \
            channels_out == self.channels_out:
            return self

        # 更新参数
        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            text_embed_dim = text_embed_dim,
            channels = channels,
            channels_out = channels_out,
            cond_on_text = cond_on_text
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    # 返回完整 Unet 配置及其参数状态字典的方法
    def to_config_and_state_dict(self):
        return self._locals, self.state_dict()

    # 从配置和状态字典中重新创建 Unet 的类方法
    @classmethod
    def from_config_and_state_dict(klass, config, state_dict):
        unet = klass(**config)
        unet.load_state_dict(state_dict)
        return unet

    # 将 Unet 持久化到磁盘的方法
    def persist_to_file(self, path):
        path = Path(path)
        path.parents[0].mkdir(exist_ok = True, parents = True)

        config, state_dict = self.to_config_and_state_dict()
        pkg = dict(config = config, state_dict = state_dict)
        torch.save(pkg, str(path))

    # 从使用 `persist_to_file` 保存的文件重新创建 Unet 的类方法
    @classmethod
    # 从文件中加载模型参数并返回实例化后的模型对象
    def hydrate_from_file(klass, path):
        # 将路径转换为 Path 对象
        path = Path(path)
        # 断言路径存在
        assert path.exists()
        # 使用 torch.load 加载模型参数
        pkg = torch.load(str(path))

        # 断言加载的参数中包含 'config' 和 'state_dict'
        assert 'config' in pkg and 'state_dict' in pkg
        # 分别获取配置和状态字典
        config, state_dict = pkg['config'], pkg['state_dict']

        # 使用配置和状态字典实例化 Unet 模型
        return Unet.from_config_and_state_dict(config, state_dict)

    # 使用分类器自由指导进行前向传播

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        # 调用 forward 方法获取 logits
        logits = self.forward(*args, **kwargs)

        # 如果 cond_scale 为 1，则直接返回 logits
        if cond_scale == 1:
            return logits

        # 使用 cond_scale 进行加权计算
        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    # 普通的前向传播方法

    def forward(
        self,
        x,
        time,
        *,
        lowres_cond_img = None,
        lowres_noise_times = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        self_cond = None,
        cond_drop_prob = 0.
# 定义一个空的 Unet 类
class NullUnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.lowres_cond = False
        self.dummy_parameter = nn.Parameter(torch.tensor([0.]))

    # 将模型参数转换为自身
    def cast_model_parameters(self, *args, **kwargs):
        return self

    # 前向传播函数，直接返回输入
    def forward(self, x, *args, **kwargs):
        return x

# 预定义的 Unet 类，配置与论文附录中的超参数对应
class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 512,
            dim_mults = (1, 2, 3, 4),
            num_resnet_blocks = 3,
            layer_attns = (False, True, True, True),
            layer_cross_attns = (False, True, True, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = False
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = (False, False, False, True),
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
            dim = 128,
            dim_mults = (1, 2, 4, 8),
            num_resnet_blocks = (2, 4, 8, 8),
            layer_attns = False,
            layer_cross_attns = (False, False, False, True),
            attn_heads = 8,
            ff_mult = 2.,
            memory_efficient = True
        )
        super().__init__(*args, **{**default_kwargs, **kwargs})

# 主要的 Imagen 类，是来自 Ho 等人的级联 DDPM
class Imagen(nn.Module):
    def __init__(
        self,
        unets,
        *,
        image_sizes,                                # 用于级联 ddpm，每个阶段的图像大小
        text_encoder_name = DEFAULT_T5_NAME,
        text_embed_dim = None,
        channels = 3,
        timesteps = 1000,
        cond_drop_prob = 0.1,
        loss_type = 'l2',
        noise_schedules = 'cosine',
        pred_objectives = 'noise',
        random_crop_sizes = None,
        lowres_noise_schedule = 'linear',
        lowres_sample_noise_level = 0.2,            # 论文中提到的一个新技巧，对低分辨率条件图像添加噪声，并在采样时将其固定到一定水平（0.1 或 0.3）- Unet 也被设计为在这��噪声水平上进行条件化
        per_sample_random_aug_noise_level = False,  # 不清楚在进行增强噪声水平条件化时，每个批次元素是否接收随机的增强噪声值-由于 @marunine 的发现，关闭此功能
        condition_on_text = True,
        auto_normalize_img = True,                  # 是否自动处理将图像从 [0, 1] 规范化为 [-1, 1] 并自动恢复-如果要自己从数据加载器传入 [-1, 1] 范围的图像，则可以关闭此功能
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.95,     # 通过查阅论文，不确定这是基于什么的
        only_train_unet_number = None,
        temporal_downsample_factor = 1,
        resize_cond_video_frames = True,
        resize_mode = 'nearest',
        min_snr_loss_weight = True,                 # https://arxiv.org/abs/2303.09556
        min_snr_gamma = 5
    def force_unconditional_(self):
        self.condition_on_text = False
        self.unconditional = True

        for unet in self.unets:
            unet.cond_on_text = False

    @property
    def device(self):
        return self._temp.device
    # 获取指定编号的 UNet 模型
    def get_unet(self, unet_number):
        # 确保编号在有效范围内
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        # 如果 self.unets 是 nn.ModuleList 类型
        if isinstance(self.unets, nn.ModuleList):
            # 将 self.unets 转换为列表
            unets_list = [unet for unet in self.unets]
            # 删除原有的 self.unets 属性
            delattr(self, 'unets')
            # 将转换后的列表重新赋值给 self.unets
            self.unets = unets_list

        # 如果指定的编号不是当前正在训练的编号
        if index != self.unet_being_trained_index:
            # 遍历所有 UNet 模型
            for unet_index, unet in enumerate(self.unets):
                # 将当前 UNet 模型移到指定设备上，其他模型移到 CPU 上
                unet.to(self.device if unet_index == index else 'cpu')

        # 更新当前正在训练的 UNet 模型编号
        self.unet_being_trained_index = index
        # 返回指定编号的 UNet 模型
        return self.unets[index]

    # 将所有 UNet 模型重置到同一设备上
    def reset_unets_all_one_device(self, device = None):
        # 设置设备为默认设备或者指定设备
        device = default(device, self.device)
        # 将所有 UNet 模型转换为 nn.ModuleList 类型
        self.unets = nn.ModuleList([*self.unets])
        # 将所有 UNet 模型移到指定设备上
        self.unets.to(device)

        # 重置当前正在训练的 UNet 模型编号
        self.unet_being_trained_index = -1

    # 使用上下文管理器将指定编号的 UNet 模型移到 GPU 上
    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        # 确保只有一个参数是有效的
        assert exists(unet_number) ^ exists(unet)

        # 如果指定了编号，则获取对应的 UNet 模型
        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        # 创建 CPU 设备
        cpu = torch.device('cpu')

        # 获取所有 UNet 模型的设备信息
        devices = [module_device(unet) for unet in self.unets]

        # 将所有 UNet 模型移到 CPU 上
        self.unets.to(cpu)
        # 将指定 UNet 模型移到当前设备上
        unet.to(self.device)

        yield

        # 将所有 UNet 模型还原到各自的设备上
        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # 重写 state_dict 函数
    def state_dict(self, *args, **kwargs):
        # 重置所有 UNet 模型到同一设备上
        self.reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    # 重写 load_state_dict 函数
    def load_state_dict(self, *args, **kwargs):
        # 重置所有 UNet 模型到同一设备上
        self.reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)

    # 高斯扩散方法

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        cond_video_frames = None,
        post_cond_video_frames = None,
        lowres_cond_img = None,
        self_cond = None,
        lowres_noise_times = None,
        cond_scale = 1.,
        model_output = None,
        t_next = None,
        pred_objective = 'noise',
        dynamic_threshold = True
    ):
        # 断言条件：如果条件为真，则抛出异常，说明不能使用分类器自由引导
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        # 初始化视频参数字典
        video_kwargs = dict()
        # 如果是视频模式，设置视频参数
        if self.is_video:
            video_kwargs = dict(
                cond_video_frames = cond_video_frames,
                post_cond_video_frames = post_cond_video_frames,
            )

        # 使用默认函数处理模型输出，获取预测结果
        pred = default(model_output, lambda: unet.forward_with_cond_scale(
            x,
            noise_scheduler.get_condition(t),
            text_embeds = text_embeds,
            text_mask = text_mask,
            cond_images = cond_images,
            cond_scale = cond_scale,
            lowres_cond_img = lowres_cond_img,
            self_cond = self_cond,
            lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_noise_times),
            **video_kwargs
        ))

        # 根据预测目标类型进行处理
        if pred_objective == 'noise':
            x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)
        elif pred_objective == 'x_start':
            x_start = pred
        elif pred_objective == 'v':
            x_start = noise_scheduler.predict_start_from_v(x, t = t, v = pred)
        else:
            raise ValueError(f'unknown objective {pred_objective}')

        # 如果启用动态阈值
        if dynamic_threshold:
            # 根据重构样本的绝对值百分位数确定动态阈值
            s = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.dynamic_thresholding_percentile,
                dim = -1
            )

            s.clamp_(min = 1.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(-1., 1.)

        # 计算均值和方差
        mean_and_variance = noise_scheduler.q_posterior(x_start = x_start, x_t = x, t = t, t_next = t_next)
        return mean_and_variance, x_start

    # 无梯度计算
    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        *,
        noise_scheduler,
        t_next = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        cond_video_frames = None,
        post_cond_video_frames = None,
        cond_scale = 1.,
        self_cond = None,
        lowres_cond_img = None,
        lowres_noise_times = None,
        pred_objective = 'noise',
        dynamic_threshold = True
    ):
        # 获取输入张量的形状和设备信息
        b, *_, device = *x.shape, x.device

        # 初始化视频参数字典
        video_kwargs = dict()
        # 如果是视频模式，设置视频参数
        if self.is_video:
            video_kwargs = dict(
                cond_video_frames = cond_video_frames,
                post_cond_video_frames = post_cond_video_frames,
            )

        # 获取均值、方差和起始值
        (model_mean, _, model_log_variance), x_start = self.p_mean_variance(
            unet,
            x = x,
            t = t,
            t_next = t_next,
            noise_scheduler = noise_scheduler,
            text_embeds = text_embeds,
            text_mask = text_mask,
            cond_images = cond_images,
            cond_scale = cond_scale,
            lowres_cond_img = lowres_cond_img,
            self_cond = self_cond,
            lowres_noise_times = lowres_noise_times,
            pred_objective = pred_objective,
            dynamic_threshold = dynamic_threshold,
            **video_kwargs
        )

        # 生成随机噪声
        noise = torch.randn_like(x)
        # 当 t == 0 时不添加噪声
        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusionContinuousTimes) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # 计算预测值
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    # 无梯度计算
    @torch.no_grad()
    # 定义一个函数 p_sample_loop，用于执行采样循环
    def p_sample_loop(
        self,
        unet,
        shape,
        *,
        noise_scheduler,
        lowres_cond_img = None,
        lowres_noise_times = None,
        text_embeds = None,
        text_mask = None,
        cond_images = None,
        cond_video_frames = None,
        post_cond_video_frames = None,
        inpaint_images = None,
        inpaint_videos = None,
        inpaint_masks = None,
        inpaint_resample_times = 5,
        init_images = None,
        skip_steps = None,
        cond_scale = 1,
        pred_objective = 'noise',
        dynamic_threshold = True,
        use_tqdm = True
    ):
        # 获取当前设备
        device = self.device

        # 获取批次大小
        batch = shape[0]
        # 生成指定形状的随机张量
        img = torch.randn(shape, device = device)

        # video

        # 判断是否为视频
        is_video = len(shape) == 5
        # 如果是视频，获取帧数
        frames = shape[-3] if is_video else None
        # 如果存在帧数，则传入目标帧数参数，否则传入空字典
        resize_kwargs = dict(target_frames = frames) if exists(frames) else dict()

        # for initialization with an image or video

        # 如果存在初始化图像
        if exists(init_images):
            # 将随机生成的图像与初始化图像相加
            img += init_images

        # keep track of x0, for self conditioning

        # 初始化 x0，用于自身条件
        x_start = None

        # prepare inpainting

        # 将 inpaint_videos 默认为 inpaint_images
        inpaint_images = default(inpaint_videos, inpaint_images)

        # 判断是否存在 inpaint_images 和 inpaint_masks
        has_inpainting = exists(inpaint_images) and exists(inpaint_masks)
        # 如果存在 inpaint_images 和 inpaint_masks，则重采样次数为 inpaint_resample_times，否则为 1
        resample_times = inpaint_resample_times if has_inpainting else 1

        # 如果存在 inpaint_images 和 inpaint_masks
        if has_inpainting:
            # 对 inpaint_images 进行归一化处理
            inpaint_images = self.normalize_img(inpaint_images)
            # 将 inpaint_images 调整大小为指定形状
            inpaint_images = self.resize_to(inpaint_images, shape[-1], **resize_kwargs)
            # 将 inpaint_masks 调整大小为指定形状，并转换为布尔类型
            inpaint_masks = self.resize_to(rearrange(inpaint_masks, 'b ... -> b 1 ...').float(), shape[-1], **resize_kwargs).bool()

        # time

        # 获取采样时间步长
        timesteps = noise_scheduler.get_sampling_timesteps(batch, device = device)

        # 是否跳过任何步骤

        # 设置默认跳过步数为 0
        skip_steps = default(skip_steps, 0)
        # 从指定步数开始采样
        timesteps = timesteps[skip_steps:]

        # video conditioning kwargs

        # 初始化视频条件参数字典
        video_kwargs = dict()
        # 如果是视频
        if self.is_video:
            # 设置视频条件参数
            video_kwargs = dict(
                cond_video_frames = cond_video_frames,
                post_cond_video_frames = post_cond_video_frames,
            )

        # 遍历时间步长
        for times, times_next in tqdm(timesteps, desc = 'sampling loop time step', total = len(timesteps), disable = not use_tqdm):
            # 判断是否为最后一个时间步长
            is_last_timestep = times_next == 0

            # 反向遍历重采样次数
            for r in reversed(range(resample_times)):
                # 判断是否为最后一个重采样步骤
                is_last_resample_step = r == 0

                # 如果存在 inpainting
                if has_inpainting:
                    # 从噪声调度器中采样噪声图像
                    noised_inpaint_images, *_ = noise_scheduler.q_sample(inpaint_images, t = times)
                    # 根据掩模进行图像修复
                    img = img * ~inpaint_masks + noised_inpaint_images * inpaint_masks

                # 如果 unet.self_cond 为真，则设置 self_cond 为 x_start，否则为 None
                self_cond = x_start if unet.self_cond else None

                # 生成图像
                img, x_start = self.p_sample(
                    unet,
                    img,
                    times,
                    t_next = times_next,
                    text_embeds = text_embeds,
                    text_mask = text_mask,
                    cond_images = cond_images,
                    cond_scale = cond_scale,
                    self_cond = self_cond,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_times = lowres_noise_times,
                    noise_scheduler = noise_scheduler,
                    pred_objective = pred_objective,
                    dynamic_threshold = dynamic_threshold,
                    **video_kwargs
                )

                # 如果存在 inpainting 且不是最后一个重采样步骤或所有时间步骤都为最后一个
                if has_inpainting and not (is_last_resample_step or torch.all(is_last_timestep)):
                    # 从指定时间点到另一个时间点采样图像
                    renoised_img = noise_scheduler.q_sample_from_to(img, times_next, times)

                    # 根据条件选择图像
                    img = torch.where(
                        self.right_pad_dims_to_datatype(is_last_timestep),
                        img,
                        renoised_img
                    )

        # 限制图像像素值范围在 -1 到 1 之间
        img.clamp_(-1., 1.)

        # final inpainting

        # 如果存在 inpainting
        if has_inpainting:
            # 根据掩模进行最终图像修复
            img = img * ~inpaint_masks + inpaint_images * inpaint_masks

        # 反归一化图像
        unnormalize_img = self.unnormalize_img(img)
        # 返回反归一化后的图像
        return unnormalize_img

    # 禁用梯度计算
    @torch.no_grad()
    # 设置评估模式装饰器
    @eval_decorator
    # 设置类型检查装饰器
    @beartype
    # 定义一个方法用于生成样本
    def sample(
        self,
        texts: List[str] = None,  # 文本列表，默认为 None
        text_masks = None,  # 文本掩码，默认为 None
        text_embeds = None,  # 文本嵌入，默认为 None
        video_frames = None,  # 视频帧，默认为 None
        cond_images = None,  # 条件图像，默认为 None
        cond_video_frames = None,  # 条件视频帧，默认为 None
        post_cond_video_frames = None,  # 后置条件视频帧，默认为 None
        inpaint_videos = None,  # 修复视频，默认为 None
        inpaint_images = None,  # 修复图像，默认为 None
        inpaint_masks = None,  # 修复掩码，默认为 None
        inpaint_resample_times = 5,  # 修复重采样次数，默认为 5
        init_images = None,  # 初始图像，默认为 None
        skip_steps = None,  # 跳过步骤，默认为 None
        batch_size = 1,  # 批量大小，默认为 1
        cond_scale = 1.,  # 条件比例，默认为 1.0
        lowres_sample_noise_level = None,  # 低分辨率采样噪声级别，默认为 None
        start_at_unet_number = 1,  # 开始于 Unet 编号，默认为 1
        start_image_or_video = None,  # 开始图像或视频，默认为 None
        stop_at_unet_number = None,  # 停止于 Unet 编号，默认为 None
        return_all_unet_outputs = False,  # 返回所有 Unet 输出，默认为 False
        return_pil_images = False,  # 返回 PIL 图像，默认为 False
        device = None,  # 设备，默认为 None
        use_tqdm = True,  # 使用 tqdm，默认为 True
        use_one_unet_in_gpu = True  # 在 GPU 中使用一个 Unet，默认为 True
    # 定义一个方法用于计算损失
    @beartype
    def p_losses(
        self,
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel],  # Unet 对象，默认为 None
        x_start,  # 起始值
        times,  # 时间
        *,
        noise_scheduler,  # 噪声调度器
        lowres_cond_img = None,  # 低分辨率条件图像，默认为 None
        lowres_aug_times = None,  # 低分辨率增强次数，默认为 None
        text_embeds = None,  # 文本嵌入，默认为 None
        text_mask = None,  # 文本掩码，默认为 None
        cond_images = None,  # 条件图像，默认为 None
        noise = None,  # 噪声，默认为 None
        times_next = None,  # 下一个时间，默认为 None
        pred_objective = 'noise',  # 预测目标，默认为 'noise'
        min_snr_gamma = None,  # 最小信噪比伽马，默认为 None
        random_crop_size = None,  # ��机裁剪大小，默认为 None
        **kwargs  # 其他关键字参数
    # 定义一个方法用于前向传播
    @beartype
    def forward(
        self,
        images,  # 图像或视频
        unet: Union[Unet, Unet3D, NullUnet, DistributedDataParallel] = None,  # Unet 对象，默认为 None
        texts: List[str] = None,  # 文本列表，默认为 None
        text_embeds = None,  # 文本嵌入，默认为 None
        text_masks = None,  # 文本掩码，默认为 None
        unet_number = None,  # Unet 编号，默认为 None
        cond_images = None,  # 条件图像，默认为 None
        **kwargs  # 其他关键字参数
```