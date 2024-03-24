# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\guided_diffusion.py`

```py
# 导入数学库
import math
# 导入拷贝库
import copy
# 导入路径库
from pathlib import Path
# 导入随机数库
from random import random
# 导入偏函数库
from functools import partial
# 导入命名元组库
from collections import namedtuple
# 导入 CPU 核心数库
from multiprocessing import cpu_count

# 导入 PyTorch 库
import torch
# 从 PyTorch 中导入神经网络模块、张量操作模块
from torch import nn, einsum
# 从 PyTorch 中导入函数操作模块
import torch.nn.functional as F
# 从 PyTorch 中导入自动混合精度模块
from torch.cuda.amp import autocast
# 从 PyTorch 中导入数据集、数据加载器
from torch.utils.data import Dataset, DataLoader

# 从 PyTorch 中导入优化器模块
from torch.optim import Adam
# 从 torchvision 中导入图像变换模块
from torchvision import transforms as T, utils

# 从 einops 中导入重排、归约函数
from einops import rearrange, reduce
# 从 einops.layers.torch 中导入重排层
from einops.layers.torch import Rearrange

# 从 PIL 中导入图像处理库
from PIL import Image
# 从 tqdm 中导入进度条库
from tqdm.auto import tqdm
# 从 ema_pytorch 中导入指数移动平均库
from ema_pytorch import EMA

# 从 accelerate 中导入加速库
from accelerate import Accelerator

# 常量

# 定义模型预测的命名元组
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# 辅助函数

# 判断变量是否存在
def exists(x):
    return x is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 返回输入值
def identity(t, *args, **kwargs):
    return t

# 无限循环数据加载器
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 判断是否存在整数平方根
def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

# 将数字转换为组
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

# 将图像标准化到 -1 到 1 之间
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 将图像反标准化到 0 到 1 之间
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# 小型辅助模块

# 残差模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

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

# RMS 归一化模块
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[-1] ** 0.5)

# 预归一化模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# 正弦位置嵌入模块
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

# 随机或学习的正弦位置嵌入模块
class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
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
    # 定义 ResNet 块的类
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        # 初始化函数，接受输入维度、输出维度、时间嵌入维度和分组数
        super().__init__()
        # 调用父类的初始化函数
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None
        # 如果存在时间嵌入维度，则创建包含激活函数和线性层的序列，否则为 None

        self.block1 = Block(dim, dim_out, groups = groups)
        # 创建第一个块，输入维度为 dim，输出维度为 dim_out，分组数为 groups
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # 创建第二个块，输入维度为 dim_out，输出维度为 dim_out，分组数为 groups
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        # 如果输入维度不等于输出维度，则创建卷积层，否则创建恒等映射

    def forward(self, x, time_emb = None):
        # 前向传播函数，接受输入 x 和时间嵌入 time_emb
        scale_shift = None
        # 初始化 scale_shift 为 None
        if exists(self.mlp) and exists(time_emb):
            # 如果存在 self.mlp 和 time_emb
            time_emb = self.mlp(time_emb)
            # 对时间嵌入进行线性变换
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # 重新排列时间嵌入的维度
            scale_shift = time_emb.chunk(2, dim = 1)
            # 将时间嵌入分成两部分，用于缩放和平移

        h = self.block1(x, scale_shift = scale_shift)
        # 使用第一个块处理输入 x，并传入缩放和平移参数

        h = self.block2(h)
        # 使用第二个块处理 h

        return h + self.res_conv(x)
        # 返回 h 与输入 x 经过卷积的结果的和

class LinearAttention(nn.Module):
    # 定义线性注意力类
    def __init__(self, dim, heads = 4, dim_head = 32):
        # 初始化函数，接受输入维度、头数和头维度
        super().__init__()
        # 调用父类的初始化函数
        self.scale = dim_head ** -0.5
        # 初始化缩放因子
        self.heads = heads
        # 头数
        hidden_dim = dim_head * heads
        # 隐藏维度为头维度乘以头数
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        # 创建用于计算查询、键、值的卷积层

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )
        # 创建输出层，包含卷积层和 RMS 归一化层

    def forward(self, x):
        # 前向传播函数，接受输入 x
        b, c, h, w = x.shape
        # 获取输入 x 的形状信息
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # 将输入 x 经过卷积层得到的结果分成查询、键、值
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        # 重新排列查询、键、值的维度

        q = q.softmax(dim = -2)
        # 对查询进行 softmax 操作
        k = k.softmax(dim = -1)
        # 对键进行 softmax 操作

        q = q * self.scale
        # 缩放查询

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        # 计算上下文信息

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        # 计算输出
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        # 重新排列输出的维度
        return self.to_out(out)
        # 返回经过输出层处理后的结果

class Attention(nn.Module):
    # 定义注意力类
    def __init__(self, dim, heads = 4, dim_head = 32):
        # 初始化函数，接受输入维度、头数和头维度
        super().__init__()
        # 调用父类的初始化函数
        self.scale = dim_head ** -0.5
        # 初始化缩放因子
        self.heads = heads
        # 头数
        hidden_dim = dim_head * heads
        # 隐藏维度为头维度乘以头数

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        # 创建用于计算查询、键、值的卷积层
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
        # 创建输出层

    def forward(self, x):
        # 前向传播函数，接受输入 x
        b, c, h, w = x.shape
        # 获取输入 x 的形状信息
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        # 将输入 x 经过卷积层得到的结果分成查询、键、值
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        # 重新排列查询、键、值的维度

        q = q * self.scale
        # 缩放查询

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        # 计算相似度
        attn = sim.softmax(dim = -1)
        # 对相似度进行 softmax 操作
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        # 计算输出
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        # 重新排列输出的维度
        return self.to_out(out)
        # 返回经过输出层处理后的结果

# model

class Unet(nn.Module):
    # 定义 Unet 类
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
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

        # 计算每一层的输入输出维度
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 部分函数应用，创建 ResnetBlock 类的实例
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
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # 创建时间 MLP 模型
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 创建下采样和上采样模块
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        # 创建中间块
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # 默认输出维度
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # 创建最终残差块和卷积层
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        # 如果需要自我条件，则拼接输入
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        # 初始卷积层
        x = self.init_conv(x)
        r = x.clone()

        # 时间嵌入
        t = self.time_mlp(time)

        h = []

        # 下采样过程
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # 中间块
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 上采样过程
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        # 最终残差块和卷积层
        x = self.final_res_block(x, t)
        return self.final_conv(x)
# 高斯扩散训练器类

# 从张量 a 中提取指定索引的值，重新形状为与 x_shape 相同的形状
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 线性 beta 调度函数，在原始 ddpm 论文中提出
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# 余弦 beta 调度函数，如 https://openreview.net/forum?id=-NEXDKk8gZ 中提出
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# S 型 beta 调度函数，如 https://arxiv.org/abs/2212.11972 中提出
def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# 高斯扩散类
class GaussianDiffusion(nn.Module):
    # 初始化函数
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_noise',
        beta_schedule='sigmoid',
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.,
        auto_normalize=True,
        min_snr_loss_weight=False,
        min_snr_gamma=5
    # 从噪声中预测起始值
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # 从起始值预测噪声
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # 预测 v
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    # 从 v 预测起始值
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    # 后验概率
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    # 根据输入的数据 x, t 以及可选的条件 x_self_cond，生成模型的预测结果
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False):
        # 使用模型对输入数据进行预测
        model_output = self.model(x, t, x_self_cond)
        # 根据是否需要对预测结果进行裁剪，选择对预测结果进行裁剪或者保持原样
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        # 根据不同的目标函数进行处理
        if self.objective == 'pred_noise':
            # 如果目标函数是预测噪声，则将模型输出作为预测噪声
            pred_noise = model_output
            # 根据预测噪声生成起始数据
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            # 可能对起始数据进行裁剪
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            # 如果目标函数是预测起始数据，则直接将模型输出作为起始数据
            x_start = model_output
            # 可能对起始数据进行裁剪
            x_start = maybe_clip(x_start)
            # 根据起始数据预测噪声
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            # 如果目标函数是预测速度，则将模型输出作为速度
            v = model_output
            # 根据速度预测起始数据
            x_start = self.predict_start_from_v(x, t, v)
            # 可能对起始数据进行裁剪
            x_start = maybe_clip(x_start)
            # 根据起始数据预测噪声
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        # 返回模型预测结果
        return ModelPrediction(pred_noise, x_start)

    # 计算模型的均值、后验方差和后验对数方差
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        # 获取模型的预测结果
        preds = self.model_predictions(x, t, x_self_cond)
        # 获取预测的起始数据
        x_start = preds.pred_x_start

        # 如果需要对起始数据进行裁剪，则进行裁剪
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # 计算模型均值、后验方差和后验对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        # 返回模型均值、后验方差、后验对数方差和起始数据
        return model_mean, posterior_variance, posterior_log_variance, x_start
     
    # 根据条件函数计算前一步的均值
    def condition_mean(self, cond_fn, mean, variance, x, t, guidance_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.
        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        # 修复官方 OpenAI 实现中的一个 bug
        # 使用前一时间步的预测均值计算梯度
        gradient = cond_fn(mean, t, **guidance_kwargs)
        # 根据梯度计算新的均值
        new_mean = (
            mean.float() + variance * gradient.float()
        )
        # 打印梯度的平均值
        print("gradient: ",(variance * gradient.float()).mean())
        # 返回新的均值
        return new_mean

    # 生成样本
    def p_sample(self, x, t: int, x_self_cond = None, cond_fn=None, guidance_kwargs=None):
        # 获取输入数据 x 的形状信息
        b, *_, device = *x.shape, x.device
        # 创建一个与 x 相同形状的张量，填充为时间步 t
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        # 获取模型的均值、方差、对数方差和起始数据
        model_mean, variance, model_log_variance, x_start = self.p_mean_variance(
            x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = True
        )
        # 如果存在条件函数和指导参数，则使用条件函数对模型均值进行条件化
        if exists(cond_fn) and exists(guidance_kwargs):
            model_mean = self.condition_mean(cond_fn, model_mean, variance, x, batched_times, guidance_kwargs)
        
        # 如果时间步大于 0，则生成噪声；否则噪声为 0
        noise = torch.randn_like(x) if t > 0 else 0.
        # 根据模型均值、对数方差和噪声生成预测图像
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        # 返回预测图像和起始数据
        return pred_img, x_start

    # 循环生成样本
    def p_sample_loop(self, shape, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        # 获取批量大小和设备信息
        batch, device = shape[0], self.betas.device

        # 创建一个随机张量作为初始图像
        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        # 在时间步上进行循环
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # 如果需要自我条件化，则使用起始数据作为条件
            self_cond = x_start if self.self_condition else None
            # 生成样本并更新起始数据
            img, x_start = self.p_sample(img, t, self_cond, cond_fn, guidance_kwargs)
            imgs.append(img)

        # 如果不需要返回所有时间步的图像，则返回最终图像
        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        # 对结果进行反归一化处理
        ret = self.unnormalize(ret)
        # 返回结果
        return ret

    # 无梯度计算
    @torch.no_grad()
    # 从给定形状中采样，返回所有时间步长的样本或者只返回最终结果
    def ddim_sample(self, shape, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        # 从形状中提取批次大小、设备、总时间步长、采样时间步长、采样率、目标
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # 在[-1, 0, 1, 2, ..., T-1]范围内生成采样时间步长+1个时间点
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))  # 将时间点倒序排列
        time_pairs = list(zip(times[:-1], times[1:]))  # 生成时间点对列表

        img = torch.randn(shape, device = device)  # 生成随机张量作为初始图像
        imgs = [img]  # 图像列表

        x_start = None  # 初始图像

        # 遍历时间点对
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)  # 生成时间条件张量
            self_cond = x_start if self.self_condition else None  # 自身条件
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)  # 获取模型预测噪声和起始图像

            imgs.append(img)  # 将图像添加到列表中

            if time_next < 0:  # 如果下一个时间点小于0
                img = x_start  # 图像更新为起始图像
                continue

            alpha = self.alphas_cumprod[time]  # 获取 alpha 值
            alpha_next = self.alphas_cumprod[time_next]  # 获取下一个 alpha 值

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()  # 计算 sigma
            c = (1 - alpha_next - sigma ** 2).sqrt()  # 计算 c

            noise = torch.randn_like(img)  # 生成噪声张量

            # 更新图像
            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)  # 返回最终图像或所有时间步长的图像序列

        ret = self.unnormalize(ret)  # 反归一化处理
        return ret  # 返回结果

    @torch.no_grad()
    def sample(self, batch_size = 16, return_all_timesteps = False, cond_fn=None, guidance_kwargs=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample  # 根据是否使用 DDIM 采样选择采样函数
        return sample_fn((batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps, cond_fn=cond_fn, guidance_kwargs=guidance_kwargs)  # 调用采样函数进行采样

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)  # 默认时间步长为总时间步长减1

        assert x1.shape == x2.shape  # 断言输入张量形状相同

        t_batched = torch.full((b,), t, device = device)  # 生成时间步长张量
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))  # 对输入张量进行采样

        img = (1 - lam) * xt1 + lam * xt2  # 插值计算

        x_start = None  # 初始图像

        # 遍历时间步长
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None  # 自身条件
            img, x_start = self.p_sample(img, i, self_cond)  # 采样图像

        return img  # 返回插值结果

    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))  # 默认噪声为随机噪声

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +  # 计算采样结果
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise  # 计算采样结果
        )
    # 定义一个函数，计算损失值
    def p_losses(self, x_start, t, noise = None):
        # 获取输入张量的形状信息
        b, c, h, w = x_start.shape
        # 如果没有提供噪声数据，则生成一个与输入张量相同形状的随机噪声张量
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 生成噪声样本

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 如果进行自条件训练，有50%的概率从当前时间集合中预测 x_start，并使用 unet 进行条件训练
        # 这种技术会使训练速度减慢 25%，但似乎显著降低 FID

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
        # 对损失进行降维处理
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 根据时间步长和损失形状调整损失权重
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    # 定义前向传播函数
    def forward(self, img, *args, **kwargs):
        # 获取输入图像的形状信息和设备信息
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言输入图像的高度和宽度必须为指定的图像大小
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # 生成随机时间步长
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 对输入图像进行归一化处理
        img = self.normalize(img)
        return self.p_losses(img, t, *args, **kwargs)
# dataset classes

# 定义一个Dataset类，继承自torch.utils.data.Dataset
class Dataset(Dataset):
    # 初始化函数，设置数据集相关参数
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        # 调用父类的初始化函数
        super().__init__()
        # 设置数据集相关参数
        self.folder = folder
        self.image_size = image_size
        # 获取指定扩展名的所有文件路径
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        # 根据是否存在图像转换函数，创建转换函数
        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        # 设置数据转换操作
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
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

# 定义一个Trainer类
class Trainer(object):
    # 初始化函数，设置训练相关参数
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        convert_image_to = None
    ):
        # 调用父类的初始化函数
        super().__init__()

        # 初始化加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        # 设置是否使用混合精度训练
        self.accelerator.native_amp = amp

        # 设置扩散模型
        self.model = diffusion_model

        # 检查样本数量是否有整数平方根
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        # 设置批量大小和梯度累积步数
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        # 创建数据集对象
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # 创建数据加载器
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        # 准备数据加载器
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        # 创建Adam优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        # 如果是主进程，创建EMA对象
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        # 创建结果文件夹
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        # 初始化步数计数器
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        # 使用加速器准备模型、数据加载器和优化器
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    # 保存模型
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        # 保存模型相关数据
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        # 将数据保存到文件
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    # 加载模型的参数和优化器状态
    def load(self, milestone):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = accelerator.device

        # 从文件中加载模型数据
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        # 获取未包装的模型
        model = self.accelerator.unwrap_model(self.model)
        # 加载模型参数
        model.load_state_dict(data['model'])

        # 加载训练步数
        self.step = data['step']
        # 加载优化器状态
        self.opt.load_state_dict(data['opt'])
        # 加载指数移动平均模型状态
        self.ema.load_state_dict(data['ema'])

        # 如果数据中包含版本信息，则打印版本信息
        if 'version' in data:
            print(f"loading from version {data['version']}")

        # 如果加速器的缩放器和数据中的缩放器存在，则加载缩放器状态
        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    # 训练模型
    def train(self):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 显示训练进度
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            # 在未达到训练步数之前循环训练
            while self.step < self.train_num_steps:

                total_loss = 0.

                # 根据梯度累积次数进行梯度累积
                for _ in range(self.gradient_accumulate_every):
                    # 获取下一个数据批次并发送到设备
                    data = next(self.dl).to(device)

                    # 使用自动混合精度计算模型损失
                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    # 反向传播
                    self.accelerator.backward(loss)

                # 对模型参数进行梯度裁剪
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 等待所有进程完成当前步骤
                accelerator.wait_for_everyone()

                # 更新优化器参数
                self.opt.step()
                self.opt.zero_grad()

                # 等待所有进程完成当前步骤
                accelerator.wait_for_everyone()

                # 更新训练步数
                self.step += 1
                # 如果是主进程
                if accelerator.is_main_process:
                    # 将指数移动平均模型发送到设备
                    self.ema.to(device)
                    # 更新指数移动平均模型
                    self.ema.update()

                    # 如果不是第一步且达到保存和采样间隔
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        # 将指数移动平均模型设置为评估模式
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            # 计算当前里程碑和批次数
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            # 生成所有图像列表
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        # 拼接所有图像
                        all_images = torch.cat(all_images_list, dim = 0)
                        # 保存图像
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                # 更新进度条
                pbar.update(1)

        # 打印训练完成信息
        accelerator.print('training complete')
if __name__ == '__main__':
    # 定义一个神经网络模型类 Classifier
    class Classifier(nn.Module):
        # 初始化函数，定义模型结构
        def __init__(self, image_size, num_classes, t_dim=1) -> None:
            super().__init__()
            # 线性层，用于处理 t 的输入
            self.linear_t = nn.Linear(t_dim, num_classes)
            # 线性层，用于处理图像的输入
            self.linear_img = nn.Linear(image_size * image_size * 3, num_classes)
        # 前向传播函数
        def forward(self, x, t):
            """
            Args:
                x (_type_): [B, 3, N, N]
                t (_type_): [B,]

            Returns:
                    logits [B, num_classes]
            """
            # 获取 batch size
            B = x.shape[0]
            # 将 t 转换为 [B, 1] 的形状
            t = t.view(B, 1)
            # 计算 logits
            logits = self.linear_t(t.float()) + self.linear_img(x.view(x.shape[0], -1))
            return logits
        
    # 定义一个函数 classifier_cond_fn，用于计算分类器输出 y 对输入 x 的梯度
    def classifier_cond_fn(x, t, classifier, y, classifier_scale=1):
        """
        return the graident of the classifier outputing y wrt x.
        formally expressed as d_log(classifier(x, t)) / dx
        """
        assert y is not None
        # 启用梯度计算
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale
            return grad
        
    # 创建 Unet 模型
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )
    image_size = 128
    # 创建 GaussianDiffusion 对象
    diffusion = GaussianDiffusion(
        model,
        image_size = image_size,
        timesteps = 1000   # number of steps
    )

    # 创建分类器对象
    classifier = Classifier(image_size=image_size, num_classes=1000, t_dim=1)
    batch_size = 4
    # 从扩散过程中采样图像
    sampled_images = diffusion.sample(
        batch_size = batch_size,
        cond_fn=classifier_cond_fn, 
        guidance_kwargs={
            "classifier":classifier,
            "y":torch.fill(torch.zeros(batch_size), 1).long(),
            "classifier_scale":1,
        }
    )
    sampled_images.shape # (4, 3, 128, 128)
```