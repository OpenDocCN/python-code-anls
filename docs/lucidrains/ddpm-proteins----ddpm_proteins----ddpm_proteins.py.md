# `.\lucidrains\ddpm-proteins\ddpm_proteins\ddpm_proteins.py`

```
import math
from math import log, pi
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image

import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat

from ddpm_proteins.utils import broadcat

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

# constants

SAVE_AND_SAMPLE_EVERY = 1000
UPDATE_EMA_EVERY = 10
EXTS = ['jpg', 'jpeg', 'png']

RESULTS_FOLDER = Path('./results')
RESULTS_FOLDER.mkdir(exist_ok = True)

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

# building block modules

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim,
        hybrid_dim_conv = False,
        groups = 8
    # 定义一个继承自 nn.Module 的类
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 定义卷积核大小和填充大小的元组
        kernels = ((3, 3),)
        paddings = ((1, 1),)

        # 如果使用混合维度卷积
        if hybrid_dim_conv:
            # 添加额外的卷积核和填充大小
            kernels = (*kernels, (9, 1), (1, 9))
            paddings = (*paddings, (4, 0), (0, 4))

        # 定义一个包含 Mish 激活函数和线性层的序列
        self.mlp = nn.Sequential(
            Mish(),
            nn.Linear(time_emb_dim, dim_out)
        )

        # 初始化输入和输出的模块列表
        self.blocks_in = nn.ModuleList([])
        self.blocks_out = nn.ModuleList([])

        # 遍历卷积核和填充大小，构建输入和输出的模块列表
        for kernel, padding in zip(kernels, paddings):
            self.blocks_in.append(nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel, padding = padding),
                nn.GroupNorm(groups, dim_out),
                Mish()
            ))

            self.blocks_out.append(nn.Sequential(
                nn.Conv2d(dim_out, dim_out, kernel, padding = padding),
                nn.GroupNorm(groups, dim_out),
                Mish()
            ))

        # 如果输入维度和输出维度不同，使用 1x1 卷积进行维度匹配，否则使用恒等映射
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 定义前向传播方法
    def forward(self, x, time_emb):
        # 对输入数据进行一系列的卷积操作，存储在 hiddens 中
        hiddens = [fn(x) for fn in self.blocks_in]

        # 对时间嵌入进行 MLP 处理
        time_emb = self.mlp(time_emb)
        # 重新排列时间嵌入的维度
        time_emb = rearrange(time_emb, 'b c -> b c () ()')

        # 将时间嵌入加到 hiddens 中
        hiddens = [h + time_emb for h in hiddens]
        # 对 hiddens 进行一系列的卷积操作
        hiddens = [fn(h) for fn, h in zip(self.blocks_out, hiddens)]
        # 将所有 hiddens 相加并加上残差连接
        return sum(hiddens) + self.res_conv(x)
# 定义应用旋转嵌入的函数
def apply_rotary_emb(q, k, pos_emb):
    # 将位置嵌入分解为正弦和余弦部分
    sin, cos = pos_emb
    dim_rotary = sin.shape[-1]
    # 分别提取旋转嵌入的正弦和余弦部分
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
    # 应用旋转嵌入到查询和键上
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    # 将处理后的查询和键重新拼接
    q, k = map(lambda t: torch.cat(t, dim = -1), ((q, q_pass), (k, k_pass)))
    return q, k

# 旋转每两个元素的函数
def rotate_every_two(x):
    # 重新排列张量形状
    x = rearrange(x, '... (d j) -> ... d j', j = 2)
    x1, x2 = x.unbind(dim = -1)
    # 旋转每两个元素
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d j -> ... (d j)')

# 定义轴向旋转嵌入类
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq = 10):
        super().__init__()
        self.dim = dim
        # 计算频率范围
        scales = torch.logspace(0., log(max_freq / 2) / log(2), self.dim // 4, base = 2)

        self.cached_pos_emb = None
        self.register_buffer('scales', scales)

    def forward(self, x):
        device, dtype, h, w = x.device, x.dtype, *x.shape[-2:]

        if exists(self.cached_pos_emb):
            return self.cached_pos_emb

        # 生成正弦和余弦位置编码
        seq_x = torch.linspace(-1., 1., steps = h, device = device)
        seq_x = seq_x.unsqueeze(-1)

        seq_y = torch.linspace(-1., 1., steps = w, device = device)
        seq_y = seq_y.unsqueeze(-1)

        scales = self.scales[(*((None,) * (len(seq_x.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        scales = self.scales[(*((None,) * (len(seq_y.shape) - 1)), Ellipsis)]
        scales = scales.to(x)

        seq_x = seq_x * scales * pi
        seq_y = seq_y * scales * pi

        x_sinu = repeat(seq_x, 'i d -> i j d', j = w)
        y_sinu = repeat(seq_y, 'j d -> i j d', i = h)

        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim = -1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim = -1)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> i j d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'i j d -> () (i j) (d r)', r = 2), (sin, cos))

        self.cached_pos_emb = (sin, cos)
        return sin, cos

# 线性注意力函数
def linear_attn_kernel(t):
    return F.elu(t) + 1

# 线性注意力机制
def linear_attention(q, k, v):
    k_sum = k.sum(dim = -2)
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_sum.type_as(q))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out

# 层归一化类
class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

# 线性注意力类
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.pos_emb = AxialRotaryEmbedding(dim = dim_head)
        self.norm = LayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = heads), (q, k, v))

        sin, cos = self.pos_emb(x)
        q, k = apply_rotary_emb(q, k, (sin, cos))

        q = linear_attn_kernel(q)
        k = linear_attn_kernel(k)
        q = q * self.scale

        out = linear_attention(q, k, v)
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

# 模型类
class Unet(nn.Module):
    # 初始化函数，设置模型参数
    def __init__(
        self,
        dim,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        groups = 8,
        channels = 3,
        condition_dim = 0,
        hybrid_dim_conv = False,

    ):
        # 调用父类的初始化函数
        super().__init__()
        self.channels = channels
        self.condition_dim = condition_dim

        # 计算输入通道数，考虑条件输入，为 MSA Transformers 做准备
        input_channels = channels + condition_dim

        # 计算不同分辨率下的维度
        dims = [input_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 初始化时间位置编码和 MLP 层
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        # 初始化下采样和上采样模块
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # 部分函数参数固定的 ResnetBlock 函数
        get_resnet_block = partial(ResnetBlock, time_emb_dim = dim, hybrid_dim_conv = hybrid_dim_conv)

        # 遍历不同分辨率下的维度
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_first = ind == 0
            is_last = ind >= (num_resolutions - 1)

            # 添加下采样模块
            self.downs.append(nn.ModuleList([
                get_resnet_block(dim_in, dim_out),
                get_resnet_block(dim_out, dim_out),
                Residual(LinearAttention(dim_out)),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        # 添加中间块
        self.mid_block1 = get_resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(LinearAttention(mid_dim))
        self.mid_block2 = get_resnet_block(mid_dim, mid_dim)

        # 遍历不同分辨率下的维度（逆序）
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            # 添加上采样模块
            self.ups.append(nn.ModuleList([
                get_resnet_block(dim_out * 2, dim_in),
                get_resnet_block(dim_in, dim_in),
                Residual(LinearAttention(dim_in)),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        # 设置输出维度
        out_dim = default(out_dim, channels)
        # 最终卷积层
        self.final_conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding = 1),
                nn.GroupNorm(groups, dim),
                Mish()
            ),
            nn.Conv2d(dim, out_dim, 1)
        )

    # 前向传播函数
    def forward(self, x, time):
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        h = []

        # 下采样过程
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 上采样过程
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
# 高斯扩散训练器类

# 从输入张量中提取指定索引的元素，返回形状与 x_shape 相同的张量
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 生成与给定形状相同的噪声张量，可以选择是否重复生成相同的噪声
def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

# 根据余弦函数生成 beta 的时间表，用于扩散过程
def cosine_beta_schedule(timesteps, s = 0.008):
    """
    余弦时间表
    参考 https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min = 0, a_max = 0.999)

# 高斯扩散模型类
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_model,
        *,
        image_size,
        timesteps = 1000,
        loss_type = 'l1',
        betas = None
    ):
        super().__init__()
        self.channels = denoise_model.channels
        self.condition_dim = denoise_model.condition_dim

        self.image_size = image_size
        self.denoise_model = denoise_model

        # 如果提供了 betas，则将其转换为 numpy 数组，否则使用余弦时间表生成 betas
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        # 将 betas、alphas_cumprod 和 alphas_cumprod_prev 注册为模型的缓冲区
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # 计算扩散 q(x_t | x_{t-1}) 和其他参数
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # 计算后验 q(x_{t-1} | x_t, x_0) 的方差
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    # 计算 q(x_t | x_{t-1}) 的均值、方差和对数方差
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # 根据噪声和当前时间步预测起始图像
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    # 计算后验分布的均值、方差和截断后的对数方差
    def q_posterior(self, x_start, x_t, t):
        # 计算后验分布的均值
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 计算后验分布的方差
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        # 获取截断后的对数方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 计算模型的均值、方差和对数方差
    def p_mean_variance(self, x, t, clip_denoised=True, condition_tensor=None):
        # 初始化去噪模型输入
        denoise_model_input = x
        # 如果存在条件张量，则将其与输入张量拼接
        if exists(condition_tensor):
            denoise_model_input = broadcat((condition_tensor, x), dim=1)

        # 使用去噪模型对输入进行去噪
        denoise_model_output = self.denoise_model(denoise_model_input, t)

        # 使用去噪后的输入预测起始值
        x_recon = self.predict_start_from_noise(x, t=t, noise=denoise_model_output)

        # 如果需要对去噪后的值进行截断
        if clip_denoised:
            x_recon.clamp_(0., 1.)

        # 计算后验分布的均值、方差和对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # 生成样本
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_tensor=None):
        # 获取输入张量的形状和设备信息
        b, *_, device = *x.shape, x.device
        # 计算模型的均值、方差和对数方差
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_tensor=condition_tensor)
        # 生成噪声
        noise = noise_like(x.shape, device, repeat_noise)
        # 当 t == 0 时不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 循环生成样本
    @torch.no_grad()
    def p_sample_loop(self, shape, condition_tensor=None):
        # 获取设备信息
        device = self.betas.device

        # 初始化图像张量
        b = shape[0]
        img = torch.randn(shape, device=device)

        # 在时间步上进行循环
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), condition_tensor=condition_tensor)

        return img

    # 生成样本
    @torch.no_grad()
    def sample(self, batch_size=16, condition_tensor=None):
        # 检查是否需要传入条件张量
        assert not (self.condition_dim > 0 and not exists(condition_tensor)), 'the conditioning tensor needs to be passed'

        # 获取图像大小和通道数，返回生成的样本
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), condition_tensor=condition_tensor)

    # 插值生成样本
    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        # 获取输入张量的形状和设备信息
        b, *_, device = *x1.shape, x1.device
        # 如果未指定时间步，则默认为最大时间步
        t = default(t, self.num_timesteps - 1)

        # 确保输入张量形状相同
        assert x1.shape == x2.shape

        # 创建时间步张量
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        # 进行插值
        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    # 从起始值生成样本
    def q_sample(self, x_start, t, noise=None):
        # 如果未提供噪声，则生成随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 使用累积平方根系数和噪声生成样本
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # 计算像素损失函数
    def p_losses(self, x_start, t, noise = None, condition_tensor = None):
        # 获取输入张量的形状信息
        b, c, h, w = x_start.shape
        # 如果没有提供噪声，则生成一个与输入张量相同形状的随机噪声张量
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 使用噪声生成噪声图像
        x_noisy = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 如果提供了条件张量，则将其与噪声图像拼接在一起
        if exists(condition_tensor):
            x_noisy = broadcat((condition_tensor, x_noisy), dim = 1)

        # 使用去噪模型对噪声图像进行去噪
        x_recon = self.denoise_model(x_noisy, t)

        # 根据损失类型计算损失值
        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        # 返回计算得到的损失值
        return loss

    # 前向传播函数
    def forward(self, x, *args, **kwargs):
        # 获取输入张量的形状信息和设备信息
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # 检查输入图像的高度和宽度是否符合要求
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # 生成随机时间步长
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        # 调用像素损失函数计算损失值并返回
        return self.p_losses(x, t, *args, **kwargs)
# dataset classes

class Dataset(data.Dataset):
    # 初始化数据集类，设置文件夹路径和图像大小
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # 获取文件夹中所有指定扩展名的文件路径
        self.paths = [p for ext in EXTS for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 设置图像转换操作
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(image_size),
            transforms.ToTensor()
        ])

    # 返回数据集的长度
    def __len__(self):
        return len(self.paths)

    # 获取指定索引处的数据
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

class Trainer(object):
    # 初始化训练器类，设置参数和数据集
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        fp16 = False,
        step_start_ema = 2000
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.step_start_ema = step_start_ema

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # 创建数据集和数据加载器
        self.ds = Dataset(folder, image_size)
        self.dl = cycle(data.DataLoader(self.ds, batch_size = train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        # 检查是否启用混合精度训练
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            # 初始化混合精度训练
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        # 重置参数
        self.reset_parameters()

    # 重置参数
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    # 更新指数移动平均模型
    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # 保存模型
    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(RESULTS_FOLDER / f'model-{milestone}.pt'))

    # 加载模型
    def load(self, milestone):
        data = torch.load(str(RESULTS_FOLDER / f'model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    # 训练模型
    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)
                print(f'{self.step}: {loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % UPDATE_EMA_EVERY == 0:
                self.step_ema()

            if self.step != 0 and self.step % SAVE_AND_SAMPLE_EVERY == 0:
                milestone = self.step // SAVE_AND_SAMPLE_EVERY
                batches = num_to_groups(36, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                utils.save_image(all_images, str(RESULTS_FOLDER / f'sample-{milestone}.png'), nrow=6)
                self.save(milestone)

            self.step += 1

        print('training completed')
```