# `.\lucidrains\recurrent-interface-network-pytorch\rin_pytorch\rin_pytorch.py`

```py
import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
from torch.special import expm1
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torchvision import transforms as T, utils

from beartype import beartype

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from rin_pytorch.attend import Attend

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator, DistributedDataParallelKwargs

# helpers functions

# 检查变量是否存在
def exists(x):
    return x is not None

# 返回输入值
def identity(x):
    return x

# 如果值存在则返回该值，否则返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 检查一个数是否可以被另一个数整除
def divisible_by(numer, denom):
    return (numer % denom) == 0

# 安全地进行除法运算
def safe_div(numer, denom, eps = 1e-10):
    return numer / denom.clamp(min = eps)

# 生成数据集的循环迭代器
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 检查一个数是否有整数平方根
def has_int_squareroot(num):
    num_sqrt = math.sqrt(num)
    return int(num_sqrt) == num_sqrt

# 将一个数分成若干组
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 将图像转换为指定类型
def convert_image_to(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 创建序列模块
def Sequential(*mods):
    return nn.Sequential(*filter(exists, mods))

# use layernorm without bias, more stable

# 自定义 LayerNorm 类
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 自定义 MultiHeadedRMSNorm 类
class MultiHeadedRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# positional embeds

# 自定义 LearnedSinusoidalPosEmb 类
class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# 自定义 LinearAttention 类
class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        norm = False,
        qk_norm = False,
        time_cond_dim = None
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.time_cond = None

        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.norm = LayerNorm(dim) if norm else nn.Identity()

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        self.to_out = nn.Sequential(
            nn.Linear(hidden_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(
        self,
        x,
        time = None
        ):
        # 获取 self.heads 的值，表示注意力头的数量
        h = self.heads
        # 对输入 x 进行归一化处理
        x = self.norm(x)

        # 如果存在时间条件
        if exists(self.time_cond):
            # 确保时间存在
            assert exists(time)
            # 将时间条件应用到输入 x 上，得到缩放和偏移量
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        # 将输入 x 转换为查询、键、值，并分成三部分
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # 如果需要对查询和键进行归一化
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 对查询和键进行 softmax 操作
        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        # 对查询结果乘以缩放因子
        q = q * self.scale

        # 计算上下文信息
        context = torch.einsum('b h n d, b h n e -> b h d e', k, v)

        # 计算输出
        out = torch.einsum('b h d e, b h n d -> b h n e', context, q)
        # 重新排列输出的维度
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 将输出传递给输出层并返回结果
        return self.to_out(out)
# 定义注意力机制模块
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        heads = 4,
        dim_head = 32,
        norm = False,
        norm_context = False,
        time_cond_dim = None,
        flash = False,
        qk_norm = False
    ):
        super().__init__()
        hidden_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.time_cond = None

        # 如果存在时间条件维度，创建时间条件模块
        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        self.scale = dim_head ** -0.5
        self.heads = heads

        # 根据是否需要归一化创建 LayerNorm 或者 nn.Identity
        self.norm = LayerNorm(dim) if norm else nn.Identity()
        self.norm_context = LayerNorm(dim_context) if norm_context else nn.Identity()

        # 创建线性变换层
        self.to_q = nn.Linear(dim, hidden_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, hidden_dim * 2, bias = False)
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

        self.qk_norm = qk_norm
        # 如果需要对 Q 和 K 进行归一化，创建 MultiHeadedRMSNorm 对象
        if qk_norm:
            self.q_norm = MultiHeadedRMSNorm(dim_head, heads)
            self.k_norm = MultiHeadedRMSNorm(dim_head, heads)

        # 创建 Attend 对象
        self.attend = Attend(flash = flash)

    def forward(
        self,
        x,
        context = None,
        time = None
    ):
        h = self.heads

        # 如果存在上下文，对上下文进行归一化
        if exists(context):
            context = self.norm_context(context)

        x = self.norm(x)

        context = default(context, x)

        # 如果存在时间条件，对输入进行时间条件处理
        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# 定义位置编码器模块
class PEG(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        # 创建深度可分离卷积层
        self.ds_conv = nn.Conv2d(dim, dim, 3, padding = 1, groups = dim)

    def forward(self, x):
        b, n, d = x.shape
        hw = int(math.sqrt(n))
        x = rearrange(x, 'b (h w) d -> b d h w', h = hw)
        x = self.ds_conv(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        return x

# 定义前馈神经网络模块
class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, time_cond_dim = None):
        super().__init__()
        self.norm = LayerNorm(dim)

        self.time_cond = None

        # 如果存在时间条件维度，创建时间条件模块
        if exists(time_cond_dim):
            self.time_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim * 2),
                Rearrange('b d -> b 1 d')
            )

            nn.init.zeros_(self.time_cond[-2].weight)
            nn.init.zeros_(self.time_cond[-2].bias)

        inner_dim = int(dim * mult)
        # 创建前馈神经网络结构
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )

    def forward(self, x, time = None):
        x = self.norm(x)

        if exists(self.time_cond):
            assert exists(time)
            scale, shift = self.time_cond(time).chunk(2, dim = -1)
            x = (x * (scale + 1)) + shift

        return self.net(x)

# 定义 RINBlock 模块
class RINBlock(nn.Module):
    def __init__(
        self,
        dim,
        latent_self_attn_depth,
        dim_latent = None,
        final_norm = True,
        patches_self_attn = True,
        **attn_kwargs
    # 初始化函数，设置模型的各个组件
    def __init__(
        self,
        dim,
        dim_latent,
        latent_self_attn_depth,
        final_norm = False,
        patches_self_attn = False,
        **attn_kwargs
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 如果未指定隐藏层维度，则使用输入维度
        dim_latent = default(dim_latent, dim)

        # 将潜在特征向量关注到补丁上的注意力机制
        self.latents_attend_to_patches = Attention(dim_latent, dim_context = dim, norm = True, norm_context = True, **attn_kwargs)
        # 潜在特征向量的交叉注意力机制和前馈网络
        self.latents_cross_attn_ff = FeedForward(dim_latent)

        # 潜在特征向量的自注意力机制列表
        self.latent_self_attns = nn.ModuleList([])
        for _ in range(latent_self_attn_depth):
            self.latent_self_attns.append(nn.ModuleList([
                Attention(dim_latent, norm = True, **attn_kwargs),
                FeedForward(dim_latent)
            ]))

        # 最终潜在特征向量的归一化层
        self.latent_final_norm = LayerNorm(dim_latent) if final_norm else nn.Identity()

        # 补丁的位置编码
        self.patches_peg = PEG(dim)
        self.patches_self_attn = patches_self_attn

        # 如果开启了补丁的自注意力机制
        if patches_self_attn:
            # 补丁的自注意力机制和前馈网络
            self.patches_self_attn = LinearAttention(dim, norm = True, **attn_kwargs)
            self.patches_self_attn_ff = FeedForward(dim)

        # 补丁关注到潜在特征向量的注意力机制和前馈网络
        self.patches_attend_to_latents = Attention(dim, dim_context = dim_latent, norm = True, norm_context = True, **attn_kwargs)
        self.patches_cross_attn_ff = FeedForward(dim)

    # 前向传播函数
    def forward(self, patches, latents, t):
        # 对补丁进行位置编码
        patches = self.patches_peg(patches) + patches

        # 潜在特征向量从补丁中提取或聚类信息
        latents = self.latents_attend_to_patches(latents, patches, time = t) + latents

        # 潜在特征向量的交叉注意力机制和前馈网络
        latents = self.latents_cross_attn_ff(latents, time = t) + latents

        # 潜在特征向量的自注意力机制
        for attn, ff in self.latent_self_attns:
            latents = attn(latents, time = t) + latents
            latents = ff(latents, time = t) + latents

        # 如果开启了补丁的自注意力机制
        if self.patches_self_attn:
            # 补丁的额外自注意力机制
            patches = self.patches_self_attn(patches, time = t) + patches
            patches = self.patches_self_attn_ff(patches) + patches

        # 补丁关注到潜在特征向量的注意力机制
        patches = self.patches_attend_to_latents(patches, latents, time = t) + patches

        # 补丁的交叉注意力机制和前馈网络
        patches = self.patches_cross_attn_ff(patches, time = t) + patches

        # 最终潜在特征向量的归一化
        latents = self.latent_final_norm(latents)
        return patches, latents
# 定义 RIN（Recursive Image Network）类，继承自 nn.Module
class RIN(nn.Module):
    # 初始化函数
    def __init__(
        self,
        dim,
        image_size,
        patch_size = 16,
        channels = 3,
        depth = 6,                      # RIN 块的数量
        latent_self_attn_depth = 2,     # 每轮从像素空间到潜在空间交叉注意力的自注意力数量
        dim_latent = None,              # 潜在空间的维度，默认为图像维度（dim）
        num_latents = 256,              # 为了获得良好结果，仍然需要使用相当数量的潜在空间（256），与 Deepmind 的 Perceiver 系列论文保持一致
        learned_sinusoidal_dim = 16,
        latent_token_time_cond = False, # 是否使用一个潜在令牌作为时间条件，或者采用自适应层归一化的方式（如其他论文“Paella” - Dominic Rampas 等所示）
        dual_patchnorm = True,
        patches_self_attn = True,       # 该存储库中的自注意力并不严格遵循论文中提出的设计。提供一种方法来移除它，以防它是不稳定的根源
        **attn_kwargs
        ):
        # 调用父类的构造函数
        super().__init__()
        # 断言图像大小能够被补丁大小整除
        assert divisible_by(image_size, patch_size)
        # 如果未指定 latent 维度，则使用默认的维度
        dim_latent = default(dim_latent, dim)

        # 设置图像大小和通道数（由于自条件，通道数乘以2）
        self.image_size = image_size
        self.channels = channels

        # 计算图像中的补丁数量和每个像素补丁的维度
        patch_height_width = image_size // patch_size
        num_patches = patch_height_width ** 2
        pixel_patch_dim = channels * (patch_size ** 2)

        # 时间条件

        # 学习的正弦位置嵌入
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        time_dim = dim * 4
        fourier_dim = learned_sinusoidal_dim + 1

        self.latent_token_time_cond = latent_token_time_cond
        time_output_dim = dim_latent if latent_token_time_cond else time_dim

        # 时间 MLP
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_output_dim)
        )

        # 像素到补丁和反向

        self.to_patches = Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(pixel_patch_dim * 2) if dual_patchnorm else None,
            nn.Linear(pixel_patch_dim * 2, dim),
            nn.LayerNorm(dim) if dual_patchnorm else None,
        )

        # 轴向位置嵌入，由 MLP 参数化

        pos_emb_dim = dim // 2

        self.axial_pos_emb_height_mlp = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, dim)
        )

        self.axial_pos_emb_width_mlp = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, pos_emb_dim),
            nn.SiLU(),
            nn.Linear(pos_emb_dim, dim)
        )

        # nn.Parameter(torch.randn(2, patch_height_width, dim) * 0.02)

        self.to_pixels = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, pixel_patch_dim),
            Rearrange('b (h w) (c p1 p2) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = patch_height_width)
        )

        # 初始化 latent
        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        nn.init.normal_(self.latents, std = 0.02)

        self.init_self_cond_latents = nn.Sequential(
            FeedForward(dim_latent),
            LayerNorm(dim_latent)
        )

        nn.init.zeros_(self.init_self_cond_latents[-1].gamma)

        # 主要的 RIN 主体参数 - 另一个注意力即可时刻

        if not latent_token_time_cond:
            attn_kwargs = {**attn_kwargs, 'time_cond_dim': time_dim}

        # 创建 RINBlock 模块列表
        self.blocks = nn.ModuleList([RINBlock(dim, dim_latent = dim_latent, latent_self_attn_depth = latent_self_attn_depth, patches_self_attn = patches_self_attn, **attn_kwargs) for _ in range(depth)])

    @property
    def device(self):
        # 返回模型参数所在的设备
        return next(self.parameters()).device

    def forward(
        self,
        x,
        time,
        x_self_cond = None,
        latent_self_cond = None,
        return_latents = False
        ):
        # 获取输入张量的批量大小
        batch = x.shape[0]

        # 如果没有给定 latents 的条件，则使用全零张量作为 latents 的条件
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))

        # 在第二维度上连接 x_self_cond 和 x，得到新的输入张量 x
        x = torch.cat((x_self_cond, x), dim = 1)

        # 准备时间条件
        t = self.time_mlp(time)

        # 准备 latents
        latents = repeat(self.latents, 'n d -> b n d', b = batch)

        # 根据论文中的方法对 latents 进行初始化
        if exists(latent_self_cond):
            latents = latents + self.init_self_cond_latents(latent_self_cond)

        # 如果将时间条件视为一个 latents token 或用于自适应层归一化的尺度和偏移
        if self.latent_token_time_cond:
            t = rearrange(t, 'b d -> b 1 d')
            latents = torch.cat((latents, t), dim = -2)

        # 将输入 x 转换为 patches
        patches = self.to_patches(x)

        # 生成高度和宽度范围
        height_range = width_range = torch.linspace(0., 1., steps = int(math.sqrt(patches.shape[-2])), device = self.device)
        pos_emb_h, pos_emb_w = self.axial_pos_emb_height_mlp(height_range), self.axial_pos_emb_width_mlp(width_range)

        # 生成位置编码
        pos_emb = rearrange(pos_emb_h, 'i d -> i 1 d') + rearrange(pos_emb_w, 'j d -> 1 j d')
        patches = patches + rearrange(pos_emb, 'i j d -> (i j) d')

        # 循环执行递归接口网络的每个块
        for block in self.blocks:
            patches, latents = block(patches, latents, t)

        # 将 patches 转换为像素
        pixels = self.to_pixels(patches)

        # 如果不需要返回 latents，则直接返回像素
        if not return_latents:
            return pixels

        # 如果设置了 latent_token_time_cond，则移除时间条件 token
        if self.latent_token_time_cond:
            latents = latents[:, :-1]

        # 返回像素和 latents
        return pixels, latents
# 定义函数，将图像归一化到[-1, 1]范围
def normalize_img(x):
    return x * 2 - 1

# 定义函数，将图像反归一化
def unnormalize_img(x):
    return (x + 1) * 0.5

# 定义函数，将带噪声图像的方差归一化，如果比例不为1
def normalize_img_variance(x, eps = 1e-5):
    std = reduce(x, 'b c h w -> b 1 1 1', partial(torch.std, unbiased = False))
    return x / std.clamp(min = eps)

# 定义函数，计算输入张量的自然对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 定义函数，将输入张量的维度右侧填充到与另一个张量相同的维度
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# 定义简单线性调度函数
def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

# 定义余弦调度函数
def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = math.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

# 定义Sigmoid调度函数
def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

# 将gamma转换为alpha和sigma
def gamma_to_alpha_sigma(gamma, scale = 1):
    return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

# 将gamma转换为对数信噪比
def gamma_to_log_snr(gamma, scale = 1, eps = 1e-5):
    return log(gamma * (scale ** 2) / (1 - gamma), eps = eps)

# 定义高斯扩散类
@beartype
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: RIN,
        *,
        timesteps = 1000,
        use_ddim = True,
        noise_schedule = 'sigmoid',
        objective = 'v',
        schedule_kwargs: dict = dict(),
        time_difference = 0.,
        min_snr_loss_weight = True,
        min_snr_gamma = 5,
        train_prob_self_cond = 0.9,
        scale = 1.                      # this will be set to < 1. for better convergence when training on higher resolution images
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels

        assert objective in {'x0', 'eps', 'v'}, 'objective must be either predict x0 or noise'
        self.objective = objective

        self.image_size = model.image_size

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        assert scale <= 1, 'scale must be less than or equal to 1'
        self.scale = scale
        self.maybe_normalize_img_variance = normalize_img_variance if scale < 1 else identity

        self.gamma_schedule = partial(self.gamma_schedule, **schedule_kwargs)

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        self.time_difference = time_difference

        self.train_prob_self_cond = train_prob_self_cond

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device
    # 获取采样时间步长
    def get_sampling_timesteps(self, batch, *, device):
        # 在设备上创建一个从1到0的等差数列，共self.timesteps+1个点
        times = torch.linspace(1., 0., self.timesteps + 1, device=device)
        # 将时间序列重复batch次
        times = repeat(times, 't -> b t', b=batch)
        # 将时间序列拆分成相邻时间对
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    # 无需梯度计算
    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference=None):
        batch, device = shape[0], self.device

        # 设置时间差值
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间对
        time_pairs = self.get_sampling_timesteps(batch, device=device)

        # 生成随机噪声图像
        img = torch.randn(shape, device=device)

        x_start = None
        last_latents = None

        # 遍历时间对
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=self.timesteps):

            # 添加时间延迟
            time_next = (time_next - self.time_difference).clamp(min=0.)

            noise_cond = time

            # 获取预测的 x0
            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, noise_cond, x_start, last_latents, return_latents=True)

            # 获取 log(snr)
            gamma = self.gamma_schedule(time)
            gamma_next = self.gamma_schedule(time_next)
            gamma, gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            # 获取 alpha 和 sigma
            alpha, sigma = gamma_to_alpha_sigma(gamma)
            alpha_next, sigma_next = gamma_to_alpha_sigma(gamma_next)

            # 计算 x0 和噪声
            if self.objective == 'x0':
                x_start = model_output
            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)
            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # 限制 x0 的取值范围
            x_start.clamp_(-1., 1.)

            # 推导后验均值和方差
            log_snr, log_snr_next = map(gamma_to_log_snr, (gamma, gamma_next))
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # 获取噪声
            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return unnormalize_img(img)

    # 无需梯度计算
    @torch.no_grad()
    # 从给定形状中获取批次和设备信息
    def ddim_sample(self, shape, time_difference = None):
        batch, device = shape[0], self.device

        # 设置时间差值为默认值或者给定值
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间步骤
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        # 生成符合正态分布的随机张量
        img = torch.randn(shape, device = device)

        x_start = None
        last_latents = None

        # 遍历时间对
        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # 获取时间和噪声水平
            gamma = self.gamma_schedule(times)
            gamma_next = self.gamma_schedule(times_next)

            # 将噪声水平填充到与图像相同的维度
            padded_gamma, padded_gamma_next = map(partial(right_pad_dims_to, img), (gamma, gamma_next))

            # 将噪声水平转换为 alpha 和 sigma
            alpha, sigma = gamma_to_alpha_sigma(padded_gamma)
            alpha_next, sigma_next = gamma_to_alpha_sigma(padded_gamma_next)

            # 添加时间延迟
            times_next = (times_next - time_difference).clamp(min = 0.)

            # 预测 x0
            maybe_normalized_img = self.maybe_normalize_img_variance(img)
            model_output, last_latents = self.model(maybe_normalized_img, times, x_start, last_latents, return_latents = True)

            # 计算 x0 和噪声
            if self.objective == 'x0':
                x_start = model_output
            elif self.objective == 'eps':
                x_start = safe_div(img - sigma * model_output, alpha)
            elif self.objective == 'v':
                x_start = alpha * img - sigma * model_output

            # 限制 x0 的取值范围
            x_start.clamp_(-1., 1.)

            # 获取预测的噪声
            pred_noise = safe_div(img - alpha * x_start, sigma)

            # 计算下一个图像
            img = x_start * alpha_next + pred_noise * sigma_next

        # 返回未归一化的图像
        return unnormalize_img(img)

    # 无需梯度计算的函数装饰器
    @torch.no_grad()
    # 生成样本
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        # 根据是否使用 DDIM 选择采样函数
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))
    # 定义一个前向传播函数，接受图像和其他参数
    def forward(self, img, *args, **kwargs):
        # 解包图像的形状和设备信息
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言图像的高度和宽度必须为指定的图像大小
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # 生成随机时间采样
        times = torch.zeros((batch,), device=device).float().uniform_(0, 1.)

        # 将图像转换为比特表示
        img = normalize_img(img)

        # 生成噪声样本
        noise = torch.randn_like(img)

        # 计算 gamma 值
        gamma = self.gamma_schedule(times)
        padded_gamma = right_pad_dims_to(img, gamma)
        alpha, sigma = gamma_to_alpha_sigma(padded_gamma, self.scale)

        # 添加噪声到图像
        noised_img = alpha * img + sigma * noise

        # 可能对图像进行归一化处理
        noised_img = self.maybe_normalize_img_variance(noised_img)

        # 在论文中，他们必须使用非常高的概率进行潜在的自我条件，高达 90% 的时间
        # 稍微有点缺点
        self_cond = self_latents = None

        if random() < self.train_prob_self_cond:
            with torch.no_grad():
                model_output, self_latents = self.model(noised_img, times, return_latents=True)
                self_latents = self_latents.detach()

                if self.objective == 'x0':
                    self_cond = model_output

                elif self.objective == 'eps':
                    self_cond = safe_div(noised_img - sigma * model_output, alpha)

                elif self.objective == 'v':
                    self_cond = alpha * noised_img - sigma * model_output

                self_cond.clamp_(-1., 1.)
                self_cond = self_cond.detach()

        # 预测并进行梯度下降步骤
        pred = self.model(noised_img, times, self_cond, self_latents)

        if self.objective == 'eps':
            target = noise

        elif self.objective == 'x0':
            target = img

        elif self.objective == 'v':
            target = alpha * noise - sigma * img

        # 计算损失
        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 最小信噪比损失权重
        snr = (alpha * alpha) / (sigma * sigma)
        maybe_clipped_snr = snr.clone()

        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=self.min_snr_gamma)

        if self.objective == 'eps':
            loss_weight = maybe_clipped_snr / snr

        elif self.objective == 'x0':
            loss_weight = maybe_clipped_snr

        elif self.objective == 'v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        return (loss * loss_weight).mean()
# dataset classes

# 定义 Dataset 类，继承自 torch.utils.data.Dataset
class Dataset(Dataset):
    # 初始化函数
    def __init__(
        self,
        folder,  # 数据集文件夹路径
        image_size,  # 图像大小
        exts = ['jpg', 'jpeg', 'png', 'tiff'],  # 图像文件扩展名列表
        augment_horizontal_flip = False,  # 是否进行水平翻转增强
        convert_image_to = None  # 图像转换函数
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        # 获取文件夹中指定扩展名的所有文件路径
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 部分应用转换函数
        maybe_convert_fn = partial(convert_image_to, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # 图像转换操作序列
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    # 返回数据集长度
    def __len__(self):
        return len(self.paths)

    # 获取指定索引处的数据
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# trainer class

# 定义 Trainer 类
@beartype
class Trainer(object):
    # 初始化函数
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,  # 扩散模型
        folder,  # 数据集文件夹路径
        *,
        train_batch_size = 16,  # 训练批量大小
        gradient_accumulate_every = 1,  # 梯度累积步数
        augment_horizontal_flip = True,  # 是否进行水平翻转增强
        train_lr = 1e-4,  # 训练学习率
        train_num_steps = 100000,  # 训练步数
        max_grad_norm = 1.,  # 梯度裁剪阈值
        ema_update_every = 10,  # EMA 更新频率
        ema_decay = 0.995,  # EMA 衰减率
        betas = (0.9, 0.99),  # Adam 优化器的 beta 参数
        save_and_sample_every = 1000,  # 保存和采样频率
        num_samples = 25,  # 采样数量
        results_folder = './results',  # 结果保存文件夹路径
        amp = False,  # 是否使用混合精度训练
        mixed_precision_type = 'fp16',  # 混合精度类型
        split_batches = True,  # 是否拆分批次
        convert_image_to = None  # 图像转换函数
    ):
        super().__init__()

        # 初始化加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no',
            kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        # 设置扩散模型
        self.model = diffusion_model

        # 检查采样数量是否有整数平方根
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # 数据集和数据加载器

        # 创建数据集对象
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # 创建数据加载器
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        # 准备数据加载器
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # 优化器

        # 创建 Adam 优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = betas)

        # 定期记录结果到文件夹

        self.results_folder = Path(results_folder)

        if self.accelerator.is_local_main_process:
            self.results_folder.mkdir(exist_ok = True)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        # 步数计数器状态

        self.step = 0

        # 准备模型、数据加载器、优化器与加速器

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    # 保存模型
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step + 1,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    # 加载指定里程碑的模型数据
    def load(self, milestone):
        # 从文件中加载模型数据
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        # 获取未加速的模型对象
        model = self.accelerator.unwrap_model(self.model)
        # 加载模型的状态字典
        model.load_state_dict(data['model'])

        # 设置当前训练步数
        self.step = data['step']
        # 加载优化器的状态字典
        self.opt.load_state_dict(data['opt'])

        # 如果是主进程，则加载指数移动平均模型的状态字典
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        # 如果加速器和数据中都存在缩放器状态字典，则加载缩放器的状态字典
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # 训练模型
    def train(self):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 显示训练进度
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            # 在未达到训练步数上限前循环训练
            while self.step < self.train_num_steps:

                total_loss = 0.

                # 根据梯度累积次数循环执行训练步骤
                for _ in range(self.gradient_accumulate_every):
                    # 获取下一个数据批次并发送到设备
                    data = next(self.dl).to(device)

                    # 使用自动混合精度计算模型损失
                    with accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    # 反向传播计算梯度
                    accelerator.backward(loss)

                # 更新进度条显示当前损失值
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 等待所有进程完成当前步骤
                accelerator.wait_for_everyone()
                # 对模型参数进行梯度裁剪
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # 执行优化器的一步更新
                self.opt.step()
                # 清空梯度
                self.opt.zero_grad()

                # 等待所有进程完成当前步骤
                accelerator.wait_for_everyone()

                # 在每个本地主进程上保存里程碑，仅在全局主进程上采样
                if accelerator.is_local_main_process:
                    milestone = self.step // self.save_and_sample_every
                    save_and_sample = self.step != 0 and self.step % self.save_and_sample_every == 0
                    
                    if accelerator.is_main_process:
                        # 将指数移动平均模型发送到设备
                        self.ema.to(device)
                        # 更新指数移动平均模型
                        self.ema.update()

                        if save_and_sample:
                            # 将指数移动平均模型设置为评估模式
                            self.ema.ema_model.eval()

                            with torch.no_grad():
                                # 将样本数量分组并生成样本图像
                                batches = num_to_groups(self.num_samples, self.batch_size)
                                all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                            all_images = torch.cat(all_images_list, dim = 0)
                            # 保存生成的样本图像
                            utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                    if save_and_sample:
                        # 保存当前里程碑的模型数据
                        self.save(milestone)

                # 更新训练步数并更新进度条
                self.step += 1
                pbar.update(1)

        # 打印训练完成信息
        accelerator.print('training complete')
```