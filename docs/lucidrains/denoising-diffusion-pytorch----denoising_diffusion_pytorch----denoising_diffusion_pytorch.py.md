# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\denoising_diffusion_pytorch.py`

```
# 导入所需的库
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__

# 定义常量
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# 辅助函数

# 检查变量是否存在
def exists(x):
    return x is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 将输入转换为元组
def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

# 检查一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 返回输入本身
def identity(t, *args, **kwargs):
    return t

# 无限循环生成数据
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 检查一个数是否有整数平方根
def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# 将一个数分成若干组
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 将图像转换为指定格式
def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 标准化函数

# 将图像像素值标准化到[-1, 1]范围
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 将标准化后的图像像素值反转回[0, 1]范围
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# 小型辅助模块

# 上采样模块
def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

# 下采样模块
def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

# RMS归一化模块
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# 正弦位置嵌入模块
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# 随机或学习的正弦位置嵌入模块
class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# 构建块模块
class Block(nn.Module):
    # 初始化函数，定义了一个卷积层、一个分组归一化层和一个SiLU激活函数
    def __init__(self, dim, dim_out, groups = 8):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个卷积层，输入维度为dim，输出维度为dim_out，卷积核大小为3，填充为1
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        # 创建一个分组归一化层，分组数为groups，输入维度为dim_out
        self.norm = nn.GroupNorm(groups, dim_out)
        # 创建一个SiLU激活函数
        self.act = nn.SiLU()
    
    # 前向传播函数，对输入进行卷积、归一化、激活操作
    def forward(self, x, scale_shift = None):
        # 对输入进行卷积操作
        x = self.proj(x)
        # 对卷积结果进行归一化操作
        x = self.norm(x)
    
        # 如果存在scale_shift参数
        if exists(scale_shift):
            # 将scale_shift参数拆分为scale和shift
            scale, shift = scale_shift
            # 对x进行缩放和平移操作
            x = x * (scale + 1) + shift
    
        # 对x进行SiLU激活操作
        x = self.act(x)
        # 返回处理后的结果
        return x
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        # 初始化 ResnetBlock 类
        super().__init__()
        # 如果存在时间嵌入维度，则创建一个包含激活函数和线性层的序列
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        # 创建两个 Block 实例
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # 如果输入维度不等于输出维度，则创建一个卷积层；否则创建一个恒等映射
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        # 初始化 scale_shift 为 None
        scale_shift = None
        # 如果存在 self.mlp 和时间嵌入，则对时间嵌入进行处理
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        # 对输入 x 应用第一个 Block
        h = self.block1(x, scale_shift = scale_shift)

        # 对 h 应用第二个 Block
        h = self.block2(h)

        # 返回 h 与输入 x 经过卷积的结果之和
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        # 初始化 LinearAttention 类
        super().__init__()
        # 初始化缩放因子和头数
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # 创建 RMSNorm 实例
        self.norm = RMSNorm(dim)

        # 初始化记忆键值对参数和转换层
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        # 创建输出转换层
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        # 获取输入 x 的形状信息
        b, c, h, w = x.shape

        # 对输入 x 进行归一化
        x = self.norm(x)

        # 将输入 x 转换为查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # 重复记忆键值对参数，���拼接到键、值中
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        # 对查询和键进行 softmax 操作
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        # 对查询进行缩放
        q = q * self.scale

        # 计算上下文信息
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # 计算输出
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        # 初始化 Attention 类
        super().__init__()
        # 初始化头数和隐藏维度
        self.heads = heads
        hidden_dim = dim_head * heads

        # 创建 RMSNorm 实例和 Attend 实例
        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        # 初始化记忆键值对参数和转换层
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        # 获取输入 x 的形状信息
        b, c, h, w = x.shape

        # 对输入 x 进行归一化
        x = self.norm(x)

        # 将输入 x 转换为查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        # 重复记忆键值对参数，并拼接到键、值中
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        # 使用 Attend 模块计算输出
        out = self.attend(q, k, v)

        # 重排输出的维度
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
        # 初始化 Unet 类，包含多个参数设置
    ):
        # 调用父类的构造函数
        super().__init__()

        # 确定维度
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        # 初始化维度
        init_dim = default(init_dim, dim)
        # 创建初始卷积层
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # 计算维度倍数
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 部分函数应用，创建 ResnetBlock 类的部分函数
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # 时间嵌入
        time_dim = dim * 4

        # 判断是否使用随机或学习的正弦位置嵌入
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            # 创建随机或学习的正弦位置嵌入对象
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            # 创建正弦位置嵌入对象
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        # 创建时间 MLP 模型
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 注意力机制
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # 层
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # 最终残差块和卷积层
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        # 返回下采样因子
        return 2 ** (len(self.downs) - 1)
    # 定义前向传播函数，接受输入 x、时间信息 time 和自身条件 x_self_cond
    def forward(self, x, time, x_self_cond = None):
        # 断言输入 x 的最后两个维度能够被 downsample_factor 整除
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        # 如果模型需要自身条件信息
        if self.self_condition:
            # 如果没有提供自身条件信息，则创建一个与 x 相同形状的全零张量
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            # 将自身条件信息与输入 x 拼接在通道维度上
            x = torch.cat((x_self_cond, x), dim = 1)

        # 初始卷积层处理输入 x
        x = self.init_conv(x)
        # 保存初始特征图
        r = x.clone()

        # 使用时间信息 t 经过时间 MLP 网络处理
        t = self.time_mlp(time)

        # 存储中间特征图的列表
        h = []

        # 遍历下采样模块列表
        for block1, block2, attn, downsample in self.downs:
            # 第一个块处理输入 x
            x = block1(x, t)
            h.append(x)  # 将处理后的特征图保存到列表中

            # 第二个块处理特征图 x
            x = block2(x, t)
            x = attn(x) + x  # 经过注意力机制后与原始特征图相加
            h.append(x)  # 将处理后的特征图保存到列表中

            # 下采样操作
            x = downsample(x)

        # 中间块处理
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        # 遍历上采样模块列表
        for block1, block2, attn, upsample in self.ups:
            # 将当前特征图与列表中的特征图拼接在通道维度上
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            # 将当前特征图与列表中的特征图拼接在通道维度上
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            # 上采样操作
            x = upsample(x)

        # 将当前特征图与初始特征图拼接在通道维度上
        x = torch.cat((x, r), dim = 1)

        # 最终残差块处理
        x = self.final_res_block(x, t)
        # 经过最终卷积层处理并返回结果
        return self.final_conv(x)
# gaussian diffusion trainer class

# 从输入张量 a 中根据索引张量 t 提取数据，返回形状与 x_shape 相同的张量
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 生成线性的 beta 衰减时间表，原始 ddpm 论文中提出
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# 生成余弦形式的 beta 衰减时间表，参考 https://openreview.net/forum?id=-NEXDKk8gZ
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# 生成 S 型形式的 beta 衰减时间表，参考 https://arxiv.org/abs/2212.11972 - Figure 8
def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# 高斯扩散模型类
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_v',
        beta_schedule='sigmoid',
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.,
        auto_normalize=True,
        offset_noise_strength=0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        min_snr_loss_weight=False,  # https://arxiv.org/abs/2303.09556
        min_snr_gamma=5
    @property
    def device(self):
        return self.betas.device

    # 根据噪声和时间步长预测起始值
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # 根据起始值和时间步长预测噪声
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # 根据起始值、时间步长和噪声预测 v 值
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    # 根据起始值、时间步长和 v 值预测起始值
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    # 计算后验概率
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    # 根据输入的数据 x, t 以及可选的条件 x_self_cond，生成模型的预测结果
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        # 使用模型生成输出
        model_output = self.model(x, t, x_self_cond)
        # 定义一个函数，根据 clip_x_start 参数决定是否对结果进行截断
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        # 根据不同的目标函数进行处理
        if self.objective == 'pred_noise':
            # 如果目标函数是预测噪声，则将模型输出作为预测噪声
            pred_noise = model_output
            # 根据预测噪声生成起始数据
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            # 如果需要对起始数据进行截断并重新计算预测噪声，则进行处理
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            # 如果目标函数是预测初始数据，则将模型输出作为初始数据
            x_start = model_output
            x_start = maybe_clip(x_start)
            # 根据初始数据预测噪声
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            # 如果目标函数是预测速度，则将模型输出作为速度
            v = model_output
            # 根据速度预测起始数据
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            # 根据起始数据预测噪声
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        # 返回模型预测结果
        return ModelPrediction(pred_noise, x_start)

    # 计算模型的均值、后验方差和后验对数方差
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        # 获取模型预测结果
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        # 如果需要对去噪后的数据进行截断，则进行处理
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # 计算模型均值、后验方差和后验对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # 生成模型的采样结果
    @torch.inference_mode()
    def p_sample(self, x, t: int, x_self_cond = None):
        # 获取输入数据的维度信息
        b, *_, device = *x.shape, self.device
        # 创建与输入数据相同维度的时间步长数据
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        # 获取模型的均值、后验方差和后验对数方差以及起始数据
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True)
        # 如果时间步长大于0，则生成噪声，否则为0
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        # 根据模型均值、后验对数方差和噪声生成预测图像
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # 循环生成模型的采样结果
    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps = False):
        # 获取批量大小和设备信息
        batch, device = shape[0], self.device

        # 生成随机初始图像
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # 循环生成采样结果
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)
            imgs.append(img)

        # 返回最终结果
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # 对结果进行反归一化处理
        ret = self.unnormalize(ret)
        return ret

    # 进入推断模式
    @torch.inference_mode()
    # 从给定形状中采样数据，可以选择返回所有时间步长的数据
    def ddim_sample(self, shape, return_all_timesteps = False):
        # 从参数中获取批量大小、设备、总时间步长、采样时间步长、采样速率、目标
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # 创建时间步长序列，[-1, 0, 1, 2, ..., T-1]，当采样时间步长等于总时间步长时
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        # 创建时间步长对，[(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))

        # 生成随机数据
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # 遍历时间步长对
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # 创建时间条件
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            # 获取模型预测结果
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True, rederive_pred_noise = True)

            # 如果下一个时间步长小于0，则更新数据并继续下一次循环
            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            # 计算 alpha 和 sigma
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            # 生成噪声数据
            noise = torch.randn_like(img)

            # 更新数据
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
            imgs.append(img)

        # 返回结果数据
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # 反归一化数据
        ret = self.unnormalize(ret)
        return ret

    # 采样函数，根据是否为 DDIM 采样选择不同的采样方式
    @torch.inference_mode()
    def sample(self, batch_size = 16, return_all_timesteps = False):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    # 插值函数，根据给定的两个数据和插值参数进行插值
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        # 遍历时间步长进行插值
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    # Q 采样函数，根据给定的起始数据、时间步长和噪声进行采样
    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # 定义一个函数，计算损失值
    def p_losses(self, x_start, t, noise = None, offset_noise_strength = None):
        # 获取输入张量的形状信息
        b, c, h, w = x_start.shape

        # 如果没有提供噪声数据，则生成一个与输入张量相同形状的随机噪声张量
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 如果存在偏移噪声强度，则添加偏移噪声
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)
        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # 生成噪声样本
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 如果进行自条件，50%的概率从当前时间集合预测 x_start，并使用 unet 进行条件
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # 预测并进行梯度下降步骤
        model_out = self.model(x, t, x_self_cond)

        # 根据不同的目标函数选择目标张量
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # 计算均方误差损失
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 根据时间步长和损失权重调整损失值
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    # 前向传播函数，对输入图像进行预处理并调用 p_losses 函数计算损失
    def forward(self, img, *args, **kwargs):
        # 获取输入图像的形状信息和设备信息
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 检查输入图像的高度和宽度是否符合要求
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # 生成随机时间步长
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 对输入图像进行归一化处理并调用 p_losses 函数计算损失
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
# dataset classes

# 定义 Dataset 类，继承自 torch.utils.data.Dataset
class Dataset(Dataset):
    # 初始化函数，设置数据集相关参数
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],  # 默认支持的图片格式
        augment_horizontal_flip = False,  # 是否进行水平翻转增强
        convert_image_to = None  # 图像转换函数，默认为空
    ):
        super().__init__()  # 调用父类的初始化函数
        self.folder = folder  # 数据集文件夹路径
        self.image_size = image_size  # 图像尺寸
        # 获取文件夹中所有指定格式的文件路径
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 如果存在图像转换函数，则使用该函数，否则使用 nn.Identity() 函数
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # 图像转换操作序列
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),  # 转换图像
            T.Resize(image_size),  # 调整图像大小
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),  # 随机水平翻转
            T.CenterCrop(image_size),  # 中心裁剪
            T.ToTensor()  # 转换为张量
        ])

    # 返回数据集的长度
    def __len__(self):
        return len(self.paths)

    # 获取指定索引处的数据
    def __getitem__(self, index):
        path = self.paths[index]  # 获取指定索引处的文件路径
        img = Image.open(path)  # 打开图像文件
        return self.transform(img)  # 返回经过转换后的图像数据

# trainer class

# 定义 Trainer 类
class Trainer(object):
    # 初始化函数，设置训练相关参数
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,  # 训练批量大小
        gradient_accumulate_every = 1,  # 梯度累积步数
        augment_horizontal_flip = True,  # 是否进行水平翻转增强
        train_lr = 1e-4,  # 训练学习率
        train_num_steps = 100000,  # 训练步数
        ema_update_every = 10,  # 指数移动平均更新频率
        ema_decay = 0.995,  # 指数移动平均衰减率
        adam_betas = (0.9, 0.99),  # Adam 优化器的 beta 参数
        save_and_sample_every = 1000,  # 保存和采样频率
        num_samples = 25,  # 采样数量
        results_folder = './results',  # 结果保存文件夹路径
        amp = False,  # 是否使用混合精度训练
        mixed_precision_type = 'fp16',  # 混合精度类型
        split_batches = True,  # 是否拆分批次
        convert_image_to = None,  # 图像转换函数
        calculate_fid = True,  # 是否计算 FID
        inception_block_idx = 2048,  # Inception 网络块索引
        max_grad_norm = 1.,  # 最大梯度范数
        num_fid_samples = 50000,  # FID 计算样本数量
        save_best_and_latest_only = False  # 是否仅保存最佳和最新结果
    ):
        # 调用父类的构造函数
        super().__init__()

        # 设置加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # 设置模型
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # 根据通道数设置默认的图像转换格式
        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # 设置采样和训练超参数
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.max_grad_norm = max_grad_norm

        # 设置数据集和数据加载器
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # 设置优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # 定期在文件夹中记录结果
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # 设置步数计数器
        self.step = 0

        # 使用加速器准备模型、数据加载器和优化器
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # 计算 FID 分数
        self.calculate_fid = calculate_fid and self.accelerator.is_main_process
        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        # 如果只保存最佳和最新结果
        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # 无穷大

        self.save_best_and_latest_only = save_best_and_latest_only

    @property
    def device(self):
        return self.accelerator.device
    # 保存模型的当前状态
    def save(self, milestone):
        # 如果不是本地主进程，则直接返回
        if not self.accelerator.is_local_main_process:
            return

        # 构建保存的数据字典
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        # 保存数据字典到文件
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    # 加载模型的状态
    def load(self, milestone):
        # 获取加速器和设备信息
        accelerator = self.accelerator
        device = accelerator.device

        # 从文件中加载数据字典
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        # 解压模型并加载状态
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        # 更新步数和优化器状态
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        
        # 如果是主进程，则加载指数移动平均模型状态
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        # 打印加载的模型版本信息
        if 'version' in data:
            print(f"loading from version {data['version']}")

        # 如果存在Scaler并且数据中也存在Scaler状态，则加载Scaler状态
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # 训练模型
    def train(self):
        # 获取加速器和设备信息
        accelerator = self.accelerator
        device = accelerator.device

        # 使用tqdm显示训练进度条
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            # 在训练步数未达到总步数之前循环训练
            while self.step < self.train_num_steps:

                total_loss = 0.

                # 梯度累积
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)

                    # 自动混合精度计算
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                # 更新进度条显示
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 等待所有进程完成
                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # 更新优化器
                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                # 更新步数
                self.step += 1
                if accelerator.is_main_process:
                    # 更新指数移动平均模型
                    self.ema.update()

                    # 每隔一定步数保存模型和生成样本
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()

                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        all_images = torch.cat(all_images_list, dim=0)

                        # 保存生成的样本图片
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))

                        # 是否计算FID
                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                # 更新进度条
                pbar.update(1)

        # 训练完成后打印信息
        accelerator.print('training complete')
```