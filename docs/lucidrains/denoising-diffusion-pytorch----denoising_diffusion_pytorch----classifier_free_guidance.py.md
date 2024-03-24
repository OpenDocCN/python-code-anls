# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\classifier_free_guidance.py`

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
import torch.nn.functional as F
from torch.cuda.amp import autocast

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

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

# 分类器无关的引导函数

# 生成指定形状的均匀分布随机数
def uniform(shape, device):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

# 生成概率掩码
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# 小型辅助模块

# 残差连接模块
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
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

# RMS归一化模块
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

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

# 构建模块
# 定义一个名为 Block 的类，继承自 nn.Module
class Block(nn.Module):
    # 初始化函数，接受 dim、dim_out 和 groups 三个参数
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        # 创建一个卷积层，输入维度为 dim，输出维度为 dim_out，卷积核大小为 3，填充为 1
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        # 创建一个 GroupNorm 层，组数为 groups，输入维度为 dim_out
        self.norm = nn.GroupNorm(groups, dim_out)
        # 创建一个 SiLU 激活函数层
        self.act = nn.SiLU()

    # 前向传播函数，接受输入 x 和 scale_shift 参数
    def forward(self, x, scale_shift = None):
        # 对输入 x 进行卷积操作
        x = self.proj(x)
        # 对卷积结果进行 GroupNorm 操作
        x = self.norm(x)

        # 如果 scale_shift 存在
        if exists(scale_shift):
            # 将 scale_shift 拆分为 scale 和 shift
            scale, shift = scale_shift
            # 对 x 进行缩放和平移操作
            x = x * (scale + 1) + shift

        # 对 x 进行 SiLU 激活函数操作
        x = self.act(x)
        # 返回处理后的 x
        return x

# 定义一个名为 ResnetBlock 的类，继承自 nn.Module
class ResnetBlock(nn.Module):
    # 初始化函数，接受 dim、dim_out、time_emb_dim、classes_emb_dim 和 groups 参数
    def __init__(self, dim, dim_out, *, time_emb_dim = None, classes_emb_dim = None, groups = 8):
        super().__init__()
        # 如果 time_emb_dim 或 classes_emb_dim 存在，则创建一个包含 SiLU 和 Linear 层的序列
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(int(time_emb_dim) + int(classes_emb_dim), dim_out * 2)
        ) if exists(time_emb_dim) or exists(classes_emb_dim) else None

        # 创建两个 Block 类实例
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # 如果 dim 不等于 dim_out，则创建一个卷积层，否则创建一个 Identity 层
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向传播函数，接受输入 x、time_emb 和 class_emb 参数
    def forward(self, x, time_emb = None, class_emb = None):

        scale_shift = None
        # 如果 self.mlp 存在且 time_emb 或 class_emb 存在
        if exists(self.mlp) and (exists(time_emb) or exists(class_emb)):
            # 将 time_emb 和 class_emb 拼接在一起
            cond_emb = tuple(filter(exists, (time_emb, class_emb)))
            cond_emb = torch.cat(cond_emb, dim = -1)
            # 将拼接后的数据传入 mlp 层
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim = 1)

        # 对输入 x 进行 Block1 处理
        h = self.block1(x, scale_shift = scale_shift)

        # 对处理后的 h 进行 Block2 处理
        h = self.block2(h)

        # 返回 h 与 x 经过 res_conv 处理后的结果相加
        return h + self.res_conv(x)

# 定义一个名为 LinearAttention 的类，继承自 nn.Module
class LinearAttention(nn.Module):
    # 初始化函数，接受 dim、heads 和 dim_head 参数
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        # 计算缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # 创建一个卷积层，用于计算 Q、K、V
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        # 创建一个序列，包含卷积层和 RMSNorm 层
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    # 前向传播函数，接受输入 x
    def forward(self, x):
        b, c, h, w = x.shape
        # 将 x 传入 QKV 卷积层，并拆分为 Q、K、V
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        # 对 Q、K 进行 softmax 操作
        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        # 计算 context
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

# 定义一个名为 Attention 的类，继承自 nn.Module
class Attention(nn.Module):
    # 初始化函数，接受 dim、heads 和 dim_head 参数
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        # 计算缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # 创建一个卷积层，用于计算 QKV
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        b, c, h, w = x.shape
        # 将 x 传入 QKV 卷积层，并拆分为 Q、K、V
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# 定义一个名为 Unet 的类，继承自 nn.Module
class Unet(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,
        num_classes,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4
    ):
        # 调用父类的构造函数
        super().__init__()

        # 分类器自由指导相关内容

        # 设置条件丢弃概率
        self.cond_drop_prob = cond_drop_prob

        # 确定维度

        # 设置通道数
        self.channels = channels
        input_channels = channels

        # 初始化维度
        init_dim = default(init_dim, dim)
        # 创建初始卷积层
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # 计算维度
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 部分函数应用，创建 ResnetBlock 类的实例
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # 时间嵌入

        # 设置时间维度
        time_dim = dim * 4

        # 判断是否使用随机或学习的正弦条件
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            # 创建随机或学习的正弦位置嵌入
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            # 创建正弦位置嵌入
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # 创建时间 MLP 模型
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 类别嵌入

        # 创建类别嵌入
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        classes_dim = dim * 4

        # 创建类别 MLP 模型
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        # 层

        # 初始化 downs 和 ups
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, dim_head = attn_dim_head, heads = attn_heads)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        # 创建最终残差块
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        rescaled_phi = 0.,
        **kwargs
    ):
        # 调用 forward 方法，传入参数 args 和 kwargs，条件概率为 0
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        # 如果条件缩放为 1，则直接返回 logits
        if cond_scale == 1:
            return logits

        # 调用 forward 方法，传入参数 args 和 kwargs，条件概率为 1
        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        # 计算缩放后的 logits
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        # 如果重新缩放的 phi 为 0，则直接返回缩放后的 logits
        if rescaled_phi == 0.:
            return scaled_logits

        # 定义计算标准差的函数
        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        # 重新缩放 logits
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        # 返回重新缩放后的 logits，根据 rescaled_phi 进行加权
        return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

    def forward(
        self,
        x,
        time,
        classes,
        cond_drop_prob = None
    ):
        # 获取输入 x 的 batch 大小和设备信息
        batch, device = x.shape[0], x.device

        # 如果未指定条件概率，则使用默认值 self.cond_drop_prob
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        # 计算类别的嵌入向量
        classes_emb = self.classes_emb(classes)

        # 如果条件概率大于 0，则进行条件概率掩码处理
        if cond_drop_prob > 0:
            # 生成保留掩码
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            # 复制 null 类别的嵌入向量，并根据掩码进行替换
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        # 计算条件信息 c
        c = self.classes_mlp(classes_emb)

        # unet 网络

        # 初始卷积层
        x = self.init_conv(x)
        r = x.clone()

        # 时间信息处理
        t = self.time_mlp(time)

        h = []

        # 下采样过程
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # 中间块处理
        x = self.mid_block1(x, t, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        # 上采样过程
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = attn(x)

            x = upsample(x)

        # 将原始输入 x 与 r 进行拼接
        x = torch.cat((x, r), dim = 1)

        # 最终残差块处理
        x = self.final_res_block(x, t, c)
        # 返回最终卷积结果
        return self.final_conv(x)
# 高斯扩散训练器类

# 从输入张量 a 中提取指定索引 t 对应的值，并根据 x_shape 的形状重新组织输出
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 线性 beta 调度函数，根据总步数 timesteps 计算出 beta 的线性变化范围
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

# 余弦 beta 调度函数，根据总步数 timesteps 和参数 s 计算出 beta 的余弦变化范围
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

# 高斯扩散模型类
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_noise',
        beta_schedule='cosine',
        ddim_sampling_eta=1.,
        offset_noise_strength=0.,
        min_snr_loss_weight=False,
        min_snr_gamma=5
        ):
        # 初始化父类
        super().__init__()
        # 断言条件，确保不是高斯扩散且模型通道数不等于模型输出维度
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        # 断言条件，确保模型的随机或学习的正弦条件为假
        assert not model.random_or_learned_sinusoidal_cond

        # 设置模型和通道数
        self.model = model
        self.channels = self.model.channels

        # 设置图像大小和目标
        self.image_size = image_size
        self.objective = objective

        # 断言条件，确保目标为预测噪声、预测图像起始或预测 v
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # 根据 beta_schedule 选择 beta 调度
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 计算 alphas 和 alphas_cumprod
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        # 设置时间步数和采样时间步数
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # 默认采样时间步数为训练时间步数

        # 断言条件，确保采样时间步数小于等于训练时间步数
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # 注册缓冲区函数，将 float64 转换为 float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # 注册缓冲区
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
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 设置偏移噪声强度和损失权重
        self.offset_noise_strength = offset_noise_strength
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max = min_snr_gamma)
        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)
        register_buffer('loss_weight', loss_weight)

    @property
    def device(self):
        return self.betas.device
    # 根据给定的输入 x_t、时间 t 和噪声 noise 预测起始值
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            # 使用累积平方根倒数系数和时间 t 提取 x_t 的部分
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            # 使用累积平方根倒数减一系数和时间 t 提取噪声 noise 的部分
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # 根据给定的输入 x_t、时间 t 和起始值 x0 预测噪声
    def predict_noise_from_start(self, x_t, t, x0):
        return (
            # 使用累积平方根倒数系数和时间 t 提取 x_t 的部分，减去起始值 x0
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            # 使用累积平方根倒数减一系数和时间 t 提取 x_t 的部分
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    # 根据给定的起始值 x_start、时间 t 和噪声 noise 预测 v
    def predict_v(self, x_start, t, noise):
        return (
            # 使用累积平方根系数和时间 t 提取噪声 noise 的部分
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            # 使用累积平方根减一系数和时间 t 提取起始值 x_start 的部分
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    # 根据给定的输入 x_t、时间 t 和 v 预测起始值
    def predict_start_from_v(self, x_t, t, v):
        return (
            # 使用累积平方根系数和时间 t 提取 x_t 的部分
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            # 使用累积平方根减一系数和时间 t 提取 v 的部分
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    # 计算后验分布的均值、方差和截断后的对数方差
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            # 使用后验均值系数1和时间 t 提取起始值 x_start 的部分
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            # 使用后验均值系数2和时间 t 提取输入 x_t 的部分
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 模型预测函数，根据不同的目标类型进行预测
    def model_predictions(self, x, t, classes, cond_scale = 6., rescaled_phi = 0.7, clip_x_start = False):
        model_output = self.model.forward_with_cond_scale(x, t, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

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
    def p_mean_variance(self, x, t, classes, cond_scale, rescaled_phi, clip_denoised = True):
        preds = self.model_predictions(x, t, classes, cond_scale, rescaled_phi)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    # 生成样本函数，根据模型预测生成图像
    @torch.no_grad()
    def p_sample(self, x, t: int, classes, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, classes = classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # 若 t == 0 则无噪声
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    # 无梯度计算装饰器
    @torch.no_grad()
    # 定义一个函数，用于生成样本的循环过程
    def p_sample_loop(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7):
        # 获取批量大小和设备信息
        batch, device = shape[0], self.betas.device

        # 生成一个符合正态分布的随机张量
        img = torch.randn(shape, device=device)

        x_start = None

        # 在时间步长上进行循环，逆序
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            # 调用 p_sample 函数生成样本
            img, x_start = self.p_sample(img, t, classes, cond_scale, rescaled_phi)

        # 将生成的图像还原到 [0, 1] 范围内
        img = unnormalize_to_zero_to_one(img)
        return img

    # 用于生成 DDIM 样本的函数
    @torch.no_grad()
    def ddim_sample(self, classes, shape, cond_scale = 6., rescaled_phi = 0.7, clip_denoised = True):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # 生成一个时间序列
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        # 生成一个符合正态分布的随机张量
        img = torch.randn(shape, device = device)

        x_start = None

        # 在时间步长上进行循环
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            # 获取模型预测的噪声和起始图像
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, classes, cond_scale = cond_scale, rescaled_phi = rescaled_phi, clip_x_start = clip_denoised)

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

        # 将生成的图像还原到 [0, 1] 范围内
        img = unnormalize_to_zero_to_one(img)
        return img

    # 生成样本的函数，根据是否使用 DDIM 采样选择不同的采样方式
    @torch.no_grad()
    def sample(self, classes, cond_scale = 6., rescaled_phi = 0.7):
        batch_size, image_size, channels = classes.shape[0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(classes, (batch_size, channels, image_size, image_size), cond_scale, rescaled_phi)

    # 插值函数，用于在两个图像之间进行插值
    @torch.no_grad()
    def interpolate(self, x1, x2, classes, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device = device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        # 在时间步长上进行循环
        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            img, _ = self.p_sample(img, i, classes)

        return img

    # 生成 q_sample 的函数，用于生成样本
    @autocast(enabled = False)
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if self.offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += self.offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # 计算像素损失函数
    def p_losses(self, x_start, t, *, classes, noise = None):
        # 获取输入张量的形状信息
        b, c, h, w = x_start.shape
        # 如果没有提供噪声，则生成一个与输入张量相同形状的随机噪声张量
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 生成噪声样本

        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 预测并进行梯度下降步骤

        model_out = self.model(x, t, classes)

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
        # 对损失进行降维操作
        loss = reduce(loss, 'b ... -> b', 'mean')

        # 根据时间步长提取损失权重并应用到损失上
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    # 前向传播函数
    def forward(self, img, *args, **kwargs):
        # 获取输入图像的形状信息和设备信息
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言输入图像的高度和宽度必须与指定的图像大小相同
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        # 在设备上生成随机时间步长
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # 将输入图像归一化到[-1, 1]范围内
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, t, *args, **kwargs)
# 示例

if __name__ == '__main__':
    # 定义类别数量
    num_classes = 10

    # 创建 Unet 模型
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        num_classes = num_classes,
        cond_drop_prob = 0.5
    )

    # 创建 GaussianDiffusion 对象
    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000
    ).cuda()

    # 创建训练图像数据
    training_images = torch.randn(8, 3, 128, 128).cuda() # 图像已经归一化为 0 到 1
    # 创建图像类别数据
    image_classes = torch.randint(0, num_classes, (8,)).cuda()    # 假设有 10 个类别

    # 计算损失
    loss = diffusion(training_images, classes = image_classes)
    loss.backward()

    # 进行多步训练

    # 生成样本图像
    sampled_images = diffusion.sample(
        classes = image_classes,
        cond_scale = 6.                # 条件缩放，大于 1 的任何值都会增强分类器的自由引导。据报道，经验上 3-8 是不错的选择
    )

    sampled_images.shape # (8, 3, 128, 128)

    # 插值

    interpolate_out = diffusion.interpolate(
        training_images[:1],
        training_images[:1],
        image_classes[:1]
    )
```