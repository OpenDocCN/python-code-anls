# `.\lucidrains\chroma-pytorch\chroma_pytorch\chroma_pytorch.py`

```
import torch  # 导入 PyTorch 库
from torch import nn, einsum  # 从 PyTorch 库中导入 nn 模块和 einsum 函数

from einops import rearrange, repeat  # 从 einops 库中导入 rearrange 和 repeat 函数

import math  # 导入 math 库
from pathlib import Path  # 从 pathlib 库中导入 Path 类
from random import random  # 从 random 库中导入 random 函数
from functools import partial  # 从 functools 库中导入 partial 函数
from multiprocessing import cpu_count  # 从 multiprocessing 库中导入 cpu_count 函数

import torch  # 重新导入 PyTorch 库
from torch import nn, einsum  # 从 PyTorch 库中重新导入 nn 模块和 einsum 函数
from torch.special import expm1  # 从 PyTorch 库中导入 expm1 函数
import torch.nn.functional as F  # 从 PyTorch 库中导入 F 模块
from torch.utils.data import Dataset, DataLoader  # 从 PyTorch 库中导入 Dataset 和 DataLoader 类

from torch.optim import Adam  # 从 PyTorch 库中导入 Adam 优化器
from torchvision import transforms as T, utils  # 从 torchvision 库中导入 transforms 模块和 utils 模块

from einops import rearrange, reduce, repeat  # 从 einops 库中重新导入 rearrange、reduce 和 repeat 函数
from einops.layers.torch import Rearrange  # 从 einops 库中导入 Rearrange 类

from tqdm.auto import tqdm  # 从 tqdm 库中导入 tqdm 函数
from ema_pytorch import EMA  # 从 ema_pytorch 库中导入 EMA 类

from accelerate import Accelerator  # 从 accelerate 库中导入 Accelerator 类

# helpers functions

def exists(x):  # 定义 exists 函数，判断变量 x 是否存在
    return x is not None

def default(val, d):  # 定义 default 函数，如果 val 存在则返回 val，否则返回 d()
    if exists(val):
        return val
    return d() if callable(d) else d

def cycle(dl):  # 定义 cycle 函数，循环生成数据集 dl 中的数据
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):  # 定义 has_int_squareroot 函数，判断 num 是否有整数平方根
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):  # 定义 num_to_groups 函数，将 num 分成 divisor 组
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to(img_type, image):  # 定义 convert_image_to 函数，将图像转换为指定类型
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# small helper modules

class Residual(nn.Module):  # 定义 Residual 类��实现残差连接
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):  # 定义 Upsample 函数，上采样操作
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):  # 定义 Downsample 函数，下采样操作
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class LayerNorm(nn.Module):  # 定义 LayerNorm 类，实现层归一化
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):  # 定义 PreNorm 类，实现预归一化
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# positional embeds

class LearnedSinusoidalPosEmb(nn.Module):  # 定义 LearnedSinusoidalPosEmb 类，实现学习的正弦位置嵌入
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

# building block modules

class Block(nn.Module):  # 定义 Block 类，实现基本块
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

class ResnetBlock(nn.Module):  # 定义 ResnetBlock 类，实现残差块
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
    # 定义前向传播函数，接受输入 x 和时间嵌入 time_emb
    def forward(self, x, time_emb = None):

        # 初始化 scale_shift 为 None
        scale_shift = None
        # 如果 self.mlp 和 time_emb 都存在
        if exists(self.mlp) and exists(time_emb):
            # 将 time_emb 输入到 self.mlp 中进行处理
            time_emb = self.mlp(time_emb)
            # 重新排列 time_emb 的维度，增加两个维度
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            # 将 time_emb 拆分成两部分，分别赋值给 scale 和 shift
            scale_shift = time_emb.chunk(2, dim = 1)

        # 将输入 x 传入第一个块中进行处理
        h = self.block1(x, scale_shift = scale_shift)

        # 将处理后的结果传入第二个块中进行处理
        h = self.block2(h)

        # 返回处理后的结果与输入 x 经过残差卷积的结果之和
        return h + self.res_conv(x)
class LinearAttention(nn.Module):
    # 定义线性注意力机制模块
    def __init__(self, dim, heads = 4, dim_head = 32):
        # 初始化函数
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # 将输入转换为查询、键、值
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            # 输出转换为指定维度
            nn.Conv2d(hidden_dim, dim, 1),
            # 对输出进行 LayerNorm 处理
            LayerNorm(dim)
        )

    def forward(self, x):
        # 前向传播函数
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    # 定义注意力机制模块
    def __init__(self, dim, heads = 4, dim_head = 32):
        # 初始化函数
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # 将输入转换为查询、键、值
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        # 前向传播函数
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class Unet(nn.Module):
    # 定义 Unet 模型
    def __init__(
        self,
        dim,
        init_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        learned_sinusoidal_dim = 16
    ):
        # 调用父类的构造函数
        super().__init__()

        # 确定维度
        self.channels = channels
        input_channels = channels * 2
        init_dim = default(init_dim, dim)
        # 初始化卷积层，输入通道数为input_channels，输出通道数为init_dim，卷积核大小为7，填充为3
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        # 计算不同层次的维度
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # 定义ResnetBlock类的部分参数
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # 时间嵌入
        time_dim = dim * 4
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        # 时间嵌入的多层感知机
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # 层次
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # 遍历不同层次的维度
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            # 添加不同层次的模块到downs列表中
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        # 中间块
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        # 反向遍历不同层次的维度
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            # 添加不同层次的模块到ups列表中
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        # 最终的残差块
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time, x_self_cond = None):

        # 默认x_self_cond为与x相同形状的全零张量
        x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
        x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        # 遍历downs列表中的模块
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

        # 遍历ups列表中的模块
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
# 定义一个名为 Chroma 的类
class Chroma(nn.Module):
    # 初始化方法
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 1000,
        use_ddim = False,
        noise_schedule = 'cosine',
        time_difference = 0.
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 设置模型和通道数
        self.model = model
        self.channels = self.model.channels

        # 设置图像大小和噪声调度
        self.image_size = image_size

        # 根据噪声调度选择不同的 log_snr 函数
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        # 设置采样时间步数和是否使用 ddim
        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # 设置时间差异
        self.time_difference = time_difference

    # 定义 device 属性
    @property
    def device(self):
        return next(self.model.parameters()).device

    # 获取采样时间步数
    def get_sampling_timesteps(self, batch, *, device):
        # 生成时间序列
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    # 生成样本
    @torch.no_grad()
    def ddpm_sample(self, shape, time_difference = None):
        # 获取 batch 大小和设备
        batch, device = shape[0], self.device

        # 设置时间差异
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间步数
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        # 生成随机噪声图像
        img = torch.randn(shape, device=device)

        x_start = None

        # 循环采样时间步数
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):

            # 添加时间延迟
            time_next = (time_next - self.time_difference).clamp(min = 0.)

            # 获取噪声条件
            noise_cond = self.log_snr(time)

            # 获取预测的 x0
            x_start = self.model(img, noise_cond, x_start)

            # 限制 x0 的范围
            x_start.clamp_(-1., 1.)

            # 获取 log(snr)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            # 获取时间和下一个时间的 alpha 和 sigma
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # 推导后验均值和方差
            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # 生成噪声
            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            # 更新图像
            img = mean + (0.5 * log_variance).exp() * noise

        return img

    @torch.no_grad()
    # 从给定形状中采样数据，可以指定时间差
    def ddim_sample(self, shape, time_difference = None):
        # 获取批次大小和设备
        batch, device = shape[0], self.device

        # 设置时间差，默认为self.time_difference
        time_difference = default(time_difference, self.time_difference)

        # 获取采样时间步
        time_pairs = self.get_sampling_timesteps(batch, device = device)

        # 生成符合正态分布的随机数据
        img = torch.randn(shape, device = device)

        x_start = None

        # 遍历时间对
        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):

            # 获取时间和噪声水平
            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            # 将噪声水平填充到与img相同的维度
            padded_log_snr, padded_log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            # 将噪声水平转换为alpha和sigma
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # 添加时间延迟
            times_next = (times_next - time_difference).clamp(min = 0.)

            # 预测x0
            x_start = self.model(img, log_snr, x_start)

            # 限制x0的取值范围
            x_start.clamp_(-1., 1.)

            # 获取预测的噪声
            pred_noise = (img - alpha * x_start) / sigma.clamp(min = 1e-8)

            # 计算下一个x
            img = x_start * alpha_next + pred_noise * sigma_next

        return img

    # 无梯度计算
    @torch.no_grad()
    def sample(self, batch_size = 16):
        image_size, channels = self.image_size, self.channels
        # 根据是否使用DDIM选择采样函数
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size))

    # 前向传播函数
    def forward(self, img, *args, **kwargs):
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言图像的高度和宽度必须为img_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # 生成随机时间
        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)

        # 生成噪声
        noise = torch.randn_like(img)

        # 获取噪声水平并填充到与img相同的维度
        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma =  log_snr_to_alpha_sigma(padded_noise_level)

        # 添加噪声到图像
        noised_img = alpha * img + sigma * noise

        # 如果进行自条件训练，50%的概率从当前时间预测x_start，并用unet进行条件
        # 这种技术会使训练速度减慢25%，但似乎显著降低FID
        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.model(noised_img, noise_level).detach_()

        # 预测并进行梯度下降
        pred = self.model(noised_img, noise_level, self_cond)

        return F.mse_loss(pred, img)
# trainer 类
class Trainer(object):
    # 初始化方法
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
        # 调用父类的初始化方法
        super().__init__()

        # 初始化加速器
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )

        # 设置是否使用 amp
        self.accelerator.native_amp = amp

        # 设置扩散模型
        self.model = diffusion_model

        # 检查 num_samples 是否有整数平方根
        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        # 设置训练批次大小和梯度累积频率
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        # 设置训练步数和图像大小
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # 数据集和数据加载器
        self.ds = Dataset(folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        # 准备数据加载器
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # 优化器
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # 定期记录结果到文件夹
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # 步数计数器状态
        self.step = 0

        # 使用加速器准备模型、数据加载器和优化器
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    # 保存模��
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    # 加载模型
    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    # 定义训练方法
    def train(self):
        # 获取加速器和设备
        accelerator = self.accelerator
        device = accelerator.device

        # 使用 tqdm 显示训练进度条，设置初始值、总步数和是否禁用
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            # 在未达到总步数前循环
            while self.step < self.train_num_steps:

                # 初始化总损失
                total_loss = 0.

                # 根据梯度累积次数循环
                for _ in range(self.gradient_accumulate_every):
                    # 获取下一个数据批次并发送到设备
                    data = next(self.dl).to(device)

                    # 使用加速器自动混合精度
                    with self.accelerator.autocast():
                        # 计算模型损失
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    # 反向传播
                    self.accelerator.backward(loss)

                # 更新进度条显示损失值
                pbar.set_description(f'loss: {total_loss:.4f}')

                # 等待所有进程完成
                accelerator.wait_for_everyone()

                # 更新优化器参数
                self.opt.step()
                self.opt.zero_grad()

                # 等待所有进程完成
                accelerator.wait_for_everyone()

                # 如果是主进程
                if accelerator.is_main_process:
                    # 将指数移动平均模型发送到设备并更新
                    self.ema.to(device)
                    self.ema.update()

                    # 如果步数不为0且可以保存和采样
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        # 将指数移动平均模型设置为评估模式
                        self.ema.ema_model.eval()

                        # 使用无梯度计算
                        with torch.no_grad():
                            # 计算里程碑和批次数
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                        # 拼接所有图像并保存
                        all_images = torch.cat(all_images_list, dim=0)
                        utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))
                        self.save(milestone)

                # 更新步数并进度条
                self.step += 1
                pbar.update(1)

        # 打印训练完成信息
        accelerator.print('training complete')
```