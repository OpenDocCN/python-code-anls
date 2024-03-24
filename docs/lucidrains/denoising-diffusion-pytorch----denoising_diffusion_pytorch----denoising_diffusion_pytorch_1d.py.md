# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\denoising_diffusion_pytorch_1d.py`

```
# 导入所需的库
import math
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

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

# 返回输入值
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

# 将图像转换为指定类型
def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# 标准化函数

# 将图像标准化到[-1, 1]范围
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 将标准化后的图像反标准化到[0, 1]范围
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# 数据集类
class Dataset1D(Dataset):
    def __init__(self, tensor: Tensor):
        super().__init__()
        self.tensor = tensor.clone()

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx].clone()

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
        nn.Conv1d(dim, default(dim_out, dim), 3, padding = 1)
    )

# 下采样模块
def Downsample(dim, dim_out = None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

# RMS归一化模块
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

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
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 重新排列输入 x 的维度，将其扩展为二维
        x = rearrange(x, 'b -> b 1')
        # 将输入 x 与权重矩阵相乘，并乘以 2π，得到频率
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        # 将频率分别计算正弦和余弦值，并拼接在一起
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        # 将原始输入 x 与计算得到的傅立叶变换结果拼接在一起
        fouriered = torch.cat((x, fouriered), dim = -1)
        # 返回拼接后的结果
        return fouriered
# 定义一个名为 Block 的类，继承自 nn.Module
class Block(nn.Module):
    # 初始化函数，接受输入维度 dim、输出维度 dim_out 和分组数 groups，默认为 8
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        # 创建一个卷积层，输入维度为 dim，输出维度为 dim_out，卷积核大小为 3，填充为 1
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        # 创建一个 Group Normalization 层，分组数为 groups，输出维度为 dim_out
        self.norm = nn.GroupNorm(groups, dim_out)
        # 创建一个 SiLU 激活函数层
        self.act = nn.SiLU()

    # 前向传播函数，接受输入 x 和可选的 scale_shift 参数
    def forward(self, x, scale_shift = None):
        # 对输入 x 进行卷积操作
        x = self.proj(x)
        # 对卷积结果进行 Group Normalization
        x = self.norm(x)

        # 如果存在 scale_shift 参数
        if exists(scale_shift):
            # 将 scale_shift 拆分为 scale 和 shift
            scale, shift = scale_shift
            # 对 x 进行缩放和平移操作
            x = x * (scale + 1) + shift

        # 对 x 进行 SiLU 激活函数操作
        x = self.act(x)
        return x

# 定义一个名为 ResnetBlock 的类，继承自 nn.Module
class ResnetBlock(nn.Module):
    # 初始化函数，接受输入维度 dim、输出维度 dim_out，时间嵌入维度 time_emb_dim（可选），分组数 groups，默认为 8
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        # 如果存在时间嵌入维度 time_emb_dim
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        # 创建两个 Block 实例，分别作用于输入和输出维度
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # 如果输入维度不等于输出维度，创建一个卷积层，否则创建一个恒等映射层
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向传播函数，接受输入 x 和时间嵌入 time_emb（可选）
    def forward(self, x, time_emb = None):

        scale_shift = None
        # 如果存在 self.mlp 和 time_emb
        if exists(self.mlp) and exists(time_emb):
            # 对时间嵌入进行线性变换
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1')
            # 将��间嵌入拆分为 scale 和 shift
            scale_shift = time_emb.chunk(2, dim = 1)

        # 对输入 x 进行第一个 Block 的操作
        h = self.block1(x, scale_shift = scale_shift)

        # 对第一个 Block 的输出进行第二个 Block 的操作
        h = self.block2(h)

        # 返回残差连接结果
        return h + self.res_conv(x)

# 定义一个名为 LinearAttention 的类，继承自 nn.Module
class LinearAttention(nn.Module):
    # 初始化函数，接受输入维度 dim、注意力头数 heads，默认为 4，注意力头维度 dim_head，默认为 32
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        # 缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # 创建一个卷积层，用于计算查询、键、值
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        # 创建输出层，包括卷积层和 RMSNorm 归一化层
        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    # 前向传播函数，接受输入 x
    def forward(self, x):
        b, c, n = x.shape
        # 将输入 x 映射为查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        # 计算注意力权重
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        # 计算上下文信息
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)
        return self.to_out(out)

# 定义一个名为 Attention 的类，继承自 nn.Module
class Attention(nn.Module):
    # 初始化函数，接受输入维度 dim、注意力头数 heads，默认为 4，注意力头维度 dim_head，默认为 32
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        # 缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # 创建一个卷积层，用于计算查询、键、值
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        # 创建输出层，只包括一个卷积层
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        b, c, n = x.shape
        # 将输入 x 映射为查询、键、值
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)

# 定义一个名为 Unet1D 的类，继承自 nn.Module
class Unet1D(nn.Module):
    # 初始化函数，接受输入维度 dim、初始维度 init_dim（可选）、输出维度 out_dim（可选）等参数
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
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        # 调用父类的构造函数
        super().__init__()

        # 确定维度
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # 初始化卷积层，输入通道数为input_channels，输出通道数为init_dim，卷积核大小为7，填充为3
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 部分函数应用，创建ResnetBlock类的部分函数，参数为resnet_block_groups
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # 时间嵌入
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            # 如果使用随机或学习的正弦位置嵌入，则创建RandomOrLearnedSinusoidalPosEmb对象
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            # 否则创建SinusoidalPosEmb对象
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        # 时间MLP模型
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 层
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv1d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
# 高斯扩散训练器类

# 从输入张量 a 中提取指定索引 t 对应的值，并根据 x_shape 的形状重新组织输出
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 线性 beta 调度函数，根据总时间步数 timesteps 计算出 beta 的线性变化范围
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# 余弦 beta 调度函数，根据总时间步数 timesteps 和参数 s 计算出 beta 的余弦变化范围
def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦调度函数
    参考 https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# 一维高斯扩散模型类
class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_noise',
        beta_schedule='cosine',
        ddim_sampling_eta=0.,
        auto_normalize=True
        ):
        # 调用父类的构造函数
        super().__init__()
        # 设置模型和通道数
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective

        # 检查目标是否合法
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # 根据 beta_schedule 选择不同的 beta 调度
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # 采样相关参数

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # 默认采样时间步数为训练时间步数

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # 注册缓冲区的辅助函数，将 float64 转换为 float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 计算扩散 q(x_t | x_{t-1}) 和其他参数

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 计算后验 q(x_{t-1} | x_t, x_0) 参数

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # 上面: 等于 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # 下面: 对 posterior_variance 进行 log 计算，因为扩散链的开始处 posterior_variance 为 0

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 计算损失权重

        snr = alphas_cumprod / (1 - alphas_cumprod)

        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

        # 是否自动归一化

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    # 根据给定的输入 x_t、时间 t 和初始值 x0，预测噪声
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # 根据给定的初始值 x_start、时间 t 和噪声，预测 v
    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    # 根据给定的输入 x_t、时间 t 和 v，预测初始值 x_start
    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    # 计算后验分布的均值、方差和截断后的对数方差
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 模型预测函数，根据不同的目标类型进行预测
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    # 计算模型的均值、方差和截断后的对数方差
    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, t, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # 生成样本，根据模型的均值和噪声生成预测图像
    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # 循环生成样本，根据给定的形状生成图像
    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond)

        img = self.unnormalize(img)
        return img

    # 禁用梯度，用于生成样本
    @torch.no_grad()
    # 从给定形状中采样，返回一个图像
    def ddim_sample(self, shape, clip_denoised = True):
        # 获取形状参数
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # 生成时间序列
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # 生成随机图像
        img = torch.randn(shape, device = device)

        x_start = None

        # 循环采样
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        # 反归一化图像
        img = self.unnormalize(img)
        return img

    # 生成样本
    @torch.no_grad()
    def sample(self, batch_size = 16):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length))

    # 插值生成图像
    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        # 插值采样
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    # 从 Q 分布中采样
    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # 定义一个函数，计算损失值
    def p_losses(self, x_start, t, noise = None):
        # 获取输入张量的形状信息
        b, c, n = x_start.shape
        # 如果没有提供噪声数据，则生成一个与输入张量相同形状的随机噪声张量
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 生成噪声样本

        # 使用给定的噪声数据生成采样结果
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 如果进行自条件训练，有50%的概率从当前时间集合中预测 x_start
        # 并使用 unet 进行条件训练
        # 这种技术会使训练速度减慢 25%，但似乎显著降低 FID
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                # 从模型预测结果中获取 x_start
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # 预测并进行梯度下降步骤

        # 使用模型进行预测
        model_out = self.model(x, t, x_self_cond)

        # 根据不同的目标函数选择目标值
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            # 预测速度 v
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # 计算均方误差损失
        loss = F.mse_loss(model_out, target, reduction = 'none')
        # 对损失进行降维处理
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 根据时间步长和损失权重对损失进行加权
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    # 前向传播函数
    def forward(self, img, *args, **kwargs):
        # 获取输入图像的形状信息和设备信息
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        # 断言输入图像序列长度与指定的序列长度相同
        assert n == seq_length, f'seq length must be {seq_length}'
        # 随机生成时间步长
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 对输入图像进行归一化处理
        img = self.normalize(img)
        # 调用 p_losses 函数计算损失值
        return self.p_losses(img, t, *args, **kwargs)
# trainer class

class Trainer1D(object):
    def __init__(
        self,
        diffusion_model: GaussianDiffusion1D,
        dataset: Dataset,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.
    ):
        super().__init__()

        # accelerator

        # 初始化加速器，根据是否使用 amp 来选择混合精度类型
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        # 设置模型和通道数
        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        # 确保 num_samples 的平方根为整数
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        # 创建数据加载器，准备数据集
        dl = DataLoader(dataset, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        # 使用 Adam 优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        # 如果是主进程，初始化 EMA 并将其移到设备上
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        # 初始化步数计数器
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        # 使用加速器准备模型、数据加载器和优化器
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    # 定义训练方法
    def train(self):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 创建进度条，显示训练进度
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            # 在未达到训练步数之前循环
            while self.step < self.train_num_steps:

                # 初始化总损失
                total_loss = 0.

                # 根据梯度累积次数循环
                for _ in range(self.gradient_accumulate_every):
                    # 从数据加载器中获取数据并发送到设备
                    data = next(self.dl).to(device)

                    # 使用加速器自动混合精度
                    with self.accelerator.autocast():
                        # 计算模型损失
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    # 反向传播
                    self.accelerator.backward(loss)

                # 更新进度条显示损失
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 等待所有进程完成
                accelerator.wait_for_everyone()
                # 对模型参数进行梯度裁剪
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # 更新优化器参数
                self.opt.step()
                self.opt.zero_grad()

                # 等待所有进程完成
                accelerator.wait_for_everyone()

                # 更新训练步数
                self.step += 1
                # 如果是主进程
                if accelerator.is_main_process:
                    # 更新指数移动平均模型
                    self.ema.update()

                    # 如果步数不为0且可以保存和采样
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        # 将指数移动平均模型设置为评估模式
                        self.ema.ema_model.eval()

                        # 使用无梯度计算
                        with torch.no_grad():
                            # 计算采样里程碑和批次
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_samples_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        # 拼接所有采样结果
                        all_samples = torch.cat(all_samples_list, dim = 0)

                        # 保存采样结果和模型
                        torch.save(all_samples, str(self.results_folder / f'sample-{milestone}.png'))
                        self.save(milestone)

                # 更新进度条
                pbar.update(1)

        # 打印训练完成信息
        accelerator.print('training complete')
```