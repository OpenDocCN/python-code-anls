# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\simple_diffusion.py`

```
# 导入数学库
import math
# 导入 functools 模块中的 partial 和 wraps 函数
from functools import partial, wraps

# 导入 torch 库
import torch
# 从 torch 库中导入 sqrt 函数
from torch import sqrt
# 从 torch 库中导入 nn 模块和 einsum 函数
from torch import nn, einsum
# 从 torch 库中导入 nn.functional 模块和 F 别名
import torch.nn.functional as F
# 从 torch.special 模块中导入 expm1 函数
from torch.special import expm1
# 从 torch.cuda.amp 模块中导入 autocast 函数

# 导入 tqdm 库
from tqdm import tqdm
# 从 einops 库中导入 rearrange、repeat、reduce、pack、unpack 函数
from einops import rearrange, repeat, reduce, pack, unpack
# 从 einops.layers.torch 模块中导入 Rearrange 类

# helpers

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 返回输入的函数
def identity(t):
    return t

# 判断是否为 lambda 函数的函数
def is_lambda(f):
    return callable(f) and f.__name__ == "<lambda>"

# 返回默认值的函数
def default(val, d):
    if exists(val):
        return val
    return d() if is_lambda(d) else d

# 将输入转换为元组的函数
def cast_tuple(t, l = 1):
    return ((t,) * l) if not isinstance(t, tuple) else t

# 在输入张量中添加维度的函数
def append_dims(t, dims):
    shape = t.shape
    return t.reshape(*shape, *((1,) * dims))

# 对输入张量进行 L2 归一化的函数
def l2norm(t):
    return F.normalize(t, dim = -1)

# u-vit 相关函数和模块

# 上采样模块
class Upsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        factor = 2
    ):
        super().__init__()
        self.factor = factor
        self.factor_squared = factor ** 2

        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * self.factor_squared, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(factor)
        )

        self.init_conv_(conv)

    # 初始化卷积层权重
    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // self.factor_squared, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o r) ...', r = self.factor_squared)

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

# 下采样模块
def Downsample(
    dim,
    dim_out = None,
    factor = 2
):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = factor, p2 = factor),
        nn.Conv2d(dim * (factor ** 2), default(dim_out, dim), 1)
    )

# RMS 归一化模块
class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, normalize_dim = 2):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim)) if scale else 1

        self.scale = scale
        self.normalize_dim = normalize_dim

    def forward(self, x):
        normalize_dim = self.normalize_dim
        scale = append_dims(self.g, x.ndim - self.normalize_dim - 1) if self.scale else 1
        return F.normalize(x, dim = normalize_dim) * scale * (x.shape[normalize_dim] ** 0.5)

# 正弦位置嵌入模块
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

# 基础模块
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    # 初始化函数，定义神经网络结构
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        # 调用父类的初始化函数
        super().__init__()
        # 如果存在时间嵌入维度，则创建包含激活函数和线性层的序列模块
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        # 创建第一个块
        self.block1 = Block(dim, dim_out, groups = groups)
        # 创建第二个块
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # 如果输入维度和输出维度不相等，则使用卷积层进行维度转换，否则使用恒等映射
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    # 前向传播函数
    def forward(self, x, time_emb = None):

        scale_shift = None
        # 如果存在时间嵌入模块和时间嵌入向量，则进行处理
        if exists(self.mlp) and exists(time_emb):
            # 对时间嵌入向量进行处理
            time_emb = self.mlp(time_emb)
            # 重新排列时间嵌入向量的维度
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # 将时间嵌入向量分成两部分，用于缩放和平移
            scale_shift = time_emb.chunk(2, dim = 1)

        # 使用第一个块处理输入数据
        h = self.block1(x, scale_shift = scale_shift)

        # 使用第二个块处理第一个块的输出
        h = self.block2(h)

        # 返回块处理后的结果与输入数据经过维度转换后的结果的和
        return h + self.res_conv(x)
class LinearAttention(nn.Module):
    # 初始化线性注意力模块
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        # 缩放因子
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        # 归一化层
        self.norm = RMSNorm(dim, normalize_dim = 1)
        # 转换输入到查询、键、值
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        # 输出转换层
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim, normalize_dim = 1)
        )

    # 前向传播函数
    def forward(self, x):
        residual = x

        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)

        return self.to_out(out) + residual

class Attention(nn.Module):
    # 初始化注意力模块
    def __init__(self, dim, heads = 4, dim_head = 32, scale = 8, dropout = 0.):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        # 归一化层
        self.norm = RMSNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)
        # 转换输入到查询、键、值
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 输出转换层
        self.to_out = nn.Linear(hidden_dim, dim, bias = False)

    # 前向传播函数
    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    # 初始化前馈神经网络模块
    def __init__(
        self,
        dim,
        cond_dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        # 归一化层
        self.norm = RMSNorm(dim, scale = False)
        dim_hidden = dim * mult

        # 缩放和偏移层
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim_hidden * 2),
            Rearrange('b d -> b 1 d')
        )

        to_scale_shift_linear = self.to_scale_shift[-2]
        nn.init.zeros_(to_scale_shift_linear.weight)
        nn.init.zeros_(to_scale_shift_linear.bias)

        # 输入投影层
        self.proj_in = nn.Sequential(
            nn.Linear(dim, dim_hidden, bias = False),
            nn.SiLU()
        )

        # 输出投影层
        self.proj_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim, bias = False)
        )

    # 前向传播函数
    def forward(self, x, t):
        x = self.norm(x)
        x = self.proj_in(x)

        scale, shift = self.to_scale_shift(t).chunk(2, dim = -1)
        x = x * (scale + 1) + shift

        return self.proj_out(x)

# vit

class Transformer(nn.Module):
    # 初始化Transformer模块
    def __init__(
        self,
        dim,
        time_cond_dim,
        depth,
        dim_head = 32,
        heads = 4,
        ff_mult = 4,
        dropout = 0.,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        # 创建多层Transformer
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim = dim, mult = ff_mult, cond_dim = time_cond_dim, dropout = dropout)
            ]))

    # 前向传播函数
    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x, t) + x

        return x
# 定义 UViT 类，继承自 nn.Module
class UViT(nn.Module):
    # 初始化函数，接受多个参数
    def __init__(
        self,
        dim,  # 特征维度
        init_dim = None,  # 初始维度，默认为 None
        out_dim = None,  # 输出维度，默认为 None
        dim_mults = (1, 2, 4, 8),  # 维度倍增因子，默认为 (1, 2, 4, 8)
        downsample_factor = 2,  # 下采样因子，默认为 2
        channels = 3,  # 通道数，默认为 3
        vit_depth = 6,  # ViT 深度，默认为 6
        vit_dropout = 0.2,  # ViT dropout 概率，默认为 0.2
        attn_dim_head = 32,  # 注意力头维度，默认为 32
        attn_heads = 4,  # 注意力头数，默认为 4
        ff_mult = 4,  # FeedForward 层倍增因子，默认为 4
        resnet_block_groups = 8,  # ResNet 块组数，默认为 8
        learned_sinusoidal_dim = 16,  # 学习的正弦维度，默认为 16
        init_img_transform: callable = None,  # 初始图像变换函数，默认为 None
        final_img_itransform: callable = None,  # 最终图像逆变换函数，默认为 None
        patch_size = 1,  # 补丁大小，默认为 1
        dual_patchnorm = False  # 双补丁规范化，默认为 False
        ):
        # 调用父类的构造函数
        super().__init__()

        # 用于初始 DWT 变换（或者研究者想要尝试的其他变换）

        if exists(init_img_transform) and exists(final_img_itransform):
            # 初始化形状为 1x1x32x32 的张量
            init_shape = torch.Size(1, 1, 32, 32)
            mock_tensor = torch.randn(init_shape)
            # 确保经过 final_img_itransform 和 init_img_transform 变换后的形状与初始形状相同
            assert final_img_itransform(init_img_transform(mock_tensor)).shape == init_shape

        # 设置初始图像变换和最终图像逆变换
        self.init_img_transform = default(init_img_transform, identity)
        self.final_img_itransform = default(final_img_itransform, identity)

        input_channels = channels

        init_dim = default(init_dim, dim)
        # 初始化卷积层，输入通道数为 input_channels，输出通道数为 init_dim，卷积核大小为 7x7，填充为 3
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # 是否进行初始补丁处理，作为 DWT 的替代方案
        self.unpatchify = identity

        input_channels = channels * (patch_size ** 2)
        needs_patch = patch_size > 1

        if needs_patch:
            if not dual_patchnorm:
                # 如果不使用双补丁规范化，则初始化卷积层
                self.init_conv = nn.Conv2d(channels, init_dim, patch_size, stride = patch_size)
            else:
                # 使用双补丁规范化
                self.init_conv = nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b h w (c p1 p2)', p1 = patch_size, p2 = patch_size),
                    nn.LayerNorm(input_channels),
                    nn.Linear(input_channels, init_dim),
                    nn.LayerNorm(init_dim),
                    Rearrange('b h w c -> b c h w')
                )

            # 反卷积层，用于将补丁还原为原始图像
            self.unpatchify = nn.ConvTranspose2d(input_channels, channels, patch_size, stride = patch_size)

        # 确定维度
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 部分 ResNet 块
        resnet_block = partial(ResnetBlock, groups = resnet_block_groups)

        # 时间嵌入
        time_dim = dim * 4

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        # 时间 MLP
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 下采样因子
        downsample_factor = cast_tuple(downsample_factor, len(dim_mults)
        assert len(downsample_factor) == len(dim_mults)

        # 层
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(in_out, downsample_factor)):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                resnet_block(dim_in, dim_in, time_emb_dim = time_dim),
                resnet_block(dim_in, dim_in, time_emb_dim = time_dim),
                LinearAttention(dim_in),
                Downsample(dim_in, dim_out, factor = factor)
            ]))

        mid_dim = dims[-1]

        # ViT 模型
        self.vit = Transformer(
            dim = mid_dim,
            time_cond_dim = time_dim,
            depth = vit_depth,
            dim_head = attn_dim_head,
            heads = attn_heads,
            ff_mult = ff_mult,
            dropout = vit_dropout
        )

        for ind, ((dim_in, dim_out), factor) in enumerate(zip(reversed(in_out), reversed(downsample_factor))):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                Upsample(dim_out, dim_in, factor = factor),
                resnet_block(dim_in * 2, dim_in, time_emb_dim = time_dim),
                resnet_block(dim_in * 2, dim_in, time_emb_dim = time_dim),
                LinearAttention(dim_in),
            ]))

        default_out_dim = input_channels
        self.out_dim = default(out_dim, default_out_dim)

        # 最终 ResNet 块和卷积层
        self.final_res_block = resnet_block(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
    # 定义前向传播函数，接受输入 x 和时间信息 time
    def forward(self, x, time):
        # 对输入图像进行初始化转换
        x = self.init_img_transform(x)

        # 初始卷积操作
        x = self.init_conv(x)
        # 保存初始特征图
        r = x.clone()

        # 时间信息通过 MLP 网络处理
        t = self.time_mlp(time)

        # 存储中间特征图的列表
        h = []

        # 下采样模块
        for block1, block2, attn, downsample in self.downs:
            # 第一个块处理
            x = block1(x, t)
            h.append(x)

            # 第二个块处理
            x = block2(x, t)
            # 注意力机制处理
            x = attn(x)
            h.append(x)

            # 下采样操作
            x = downsample(x)

        # 重新排列特征图维度
        x = rearrange(x, 'b c h w -> b h w c')
        # 打包特征图
        x, ps = pack([x], 'b * c')

        # Vision Transformer 处理
        x = self.vit(x, t)

        # 解包特征图
        x, = unpack(x, ps, 'b * c')
        # 重新排列特征图维度
        x = rearrange(x, 'b h w c -> b c h w')

        # 上采样模块
        for upsample, block1, block2, attn in self.ups:
            # 上采样操作
            x = upsample(x)

            # 拼接特征图
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            # 拼接特征图
            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

        # 拼接初始特征图
        x = torch.cat((x, r), dim = 1)

        # 最终残差块处理
        x = self.final_res_block(x, t)
        # 最终卷积操作
        x = self.final_conv(x)

        # 反向解除图像补丁
        x = self.unpatchify(x)
        # 返回最终图像
        return self.final_img_itransform(x)
# normalization functions

# 将图像数据归一化到 [-1, 1] 范围
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 将归一化后的数据反归一化到 [0, 1] 范围
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

# 将 t 张量的维度右侧填充到与 x 张量相同维度
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# logsnr schedules and shifting / interpolating decorators
# only cosine for now

# 计算张量 t 的对数，避免 t 小于 eps 时取对数出错
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 计算 logsnr 的余弦调度
def logsnr_schedule_cosine(t, logsnr_min = -15, logsnr_max = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))

# 对 logsnr_schedule_cosine 进行偏移
def logsnr_schedule_shifted(fn, image_d, noise_d):
    shift = 2 * math.log(noise_d / image_d)
    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal shift
        return fn(*args, **kwargs) + shift
    return inner

# 对 logsnr_schedule_cosine 进行插值
def logsnr_schedule_interpolated(fn, image_d, noise_d_low, noise_d_high):
    logsnr_low_fn = logsnr_schedule_shifted(fn, image_d, noise_d_low)
    logsnr_high_fn = logsnr_schedule_shifted(fn, image_d, noise_d_high)

    @wraps(fn)
    def inner(t, *args, **kwargs):
        nonlocal logsnr_low_fn
        nonlocal logsnr_high_fn
        return t * logsnr_low_fn(t, *args, **kwargs) + (1 - t) * logsnr_high_fn(t, *args, **kwargs)

    return inner

# main gaussian diffusion class

# 高斯扩散类
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model: UViT,
        *,
        image_size,
        channels = 3,
        pred_objective = 'v',
        noise_schedule = logsnr_schedule_cosine,
        noise_d = None,
        noise_d_low = None,
        noise_d_high = None,
        num_sample_steps = 500,
        clip_sample_denoised = True,
        min_snr_loss_weight = True,
        min_snr_gamma = 5
    ):
        super().__init__()
        assert pred_objective in {'v', 'eps'}, 'whether to predict v-space (progressive distillation paper) or noise'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # training objective

        self.pred_objective = pred_objective

        # noise schedule

        assert not all([*map(exists, (noise_d, noise_d_low, noise_d_high))]), 'you must either set noise_d for shifted schedule, or noise_d_low and noise_d_high for shifted and interpolated schedule'

        # determine shifting or interpolated schedules

        self.log_snr = noise_schedule

        if exists(noise_d):
            self.log_snr = logsnr_schedule_shifted(self.log_snr, image_size, noise_d)

        if exists(noise_d_low) or exists(noise_d_high):
            assert exists(noise_d_low) and exists(noise_d_high), 'both noise_d_low and noise_d_high must be set'

            self.log_snr = logsnr_schedule_interpolated(self.log_snr, image_size, noise_d_low, noise_d_high)

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # loss weight

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        return next(self.model.parameters()).device
    # 计算均值和方差
    def p_mean_variance(self, x, time, time_next):
        
        # 计算当前时间点和下一个时间点的对数信噪比
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        # 计算 c 值
        c = -expm1(log_snr - log_snr_next)

        # 计算 alpha 和 sigma
        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()
        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        # 重复 log_snr 以匹配 x 的形状
        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        # 使用模型预测
        pred = self.model(x, batch_log_snr)

        # 根据预测目标选择不同的计算方式
        if self.pred_objective == 'v':
            x_start = alpha * x - sigma * pred
        elif self.pred_objective == 'eps':
            x_start = (x - sigma * pred) / alpha

        # 将 x_start 限制在 -1 到 1 之间
        x_start.clamp_(-1., 1.)

        # 计算模型均值和后验方差
        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # 采样相关函数

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device

        # 计算模型均值和方差
        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        # 如果是最后一个时间点，则直接返回模型均值
        if time_next == 0:
            return model_mean

        # 生成噪声并返回采样结果
        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]

        # 生成随机初始图像
        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        # 循环进行采样
        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        # 将图像限制在 -1 到 1 之间，并反归一化到 [0, 1] 范围
        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # 训练相关函数 - 噪声预测

    @autocast(enabled = False)
    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 计算 alpha 和 sigma，生成带噪声的图像
        log_snr = self.log_snr(times)
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr

    # 计算损失函数
    def p_losses(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 生成带噪声的图像并计算模型输出
        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        model_out = self.model(x, log_snr)

        # 根据预测目标选择不同的计算方式
        if self.pred_objective == 'v':
            padded_log_snr = right_pad_dims_to(x, log_snr)
            alpha, sigma = padded_log_snr.sigmoid().sqrt(), (-padded_log_snr).sigmoid().sqrt()
            target = alpha * noise - sigma * x_start
        elif self.pred_objective == 'eps':
            target = noise

        # 计算均方误差损失
        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        snr = log_snr.exp()

        maybe_clip_snr = snr.clone()
        if self.min_snr_loss_weight:
            maybe_clip_snr.clamp_(max = self.min_snr_gamma)

        # 根据预测目标选择不同的损失权重计算方式
        if self.pred_objective == 'v':
            loss_weight = maybe_clip_snr / (snr + 1)
        elif self.pred_objective == 'eps':
            loss_weight = maybe_clip_snr / snr

        return (loss * loss_weight).mean()
    # 定义一个前向传播函数，接受图像和其他参数
    def forward(self, img, *args, **kwargs):
        # 解包图像的形状信息，包括通道数、高度、宽度等
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言图像的高度和宽度必须等于指定的图像大小
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # 将图像数据归一化到 -1 到 1 之间
        img = normalize_to_neg_one_to_one(img)
        # 创建一个与图像数量相同的随机时间数组
        times = torch.zeros((img.shape[0],), device = self.device).float().uniform_(0, 1)

        # 调用损失函数计算函数，传入图像、时间和其他参数
        return self.p_losses(img, times, *args, **kwargs)
```