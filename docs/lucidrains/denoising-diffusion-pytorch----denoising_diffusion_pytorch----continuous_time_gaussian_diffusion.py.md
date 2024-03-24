# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\continuous_time_gaussian_diffusion.py`

```py
import math
import torch
from torch import sqrt
from torch import nn, einsum
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.special import expm1

from tqdm import tqdm
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers

# 检查值是否存在
def exists(val):
    return val is not None

# 返回值或默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# normalization functions

# 将图像归一化到[-1, 1]范围
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 将张量反归一化到[0, 1]范围
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

# 将t的维度右侧填充到与x相同维度
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# neural net helpers

# 残差模块
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)

# 单调线性模块
class MonotonicLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return F.linear(x, self.net.weight.abs(), self.net.bias.abs())

# continuous schedules

# 基于论文中的公式，定义log(snr)函数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 基于线性log(snr)的beta函数
def beta_linear_log_snr(t):
    return -log(expm1(1e-4 + 10 * (t ** 2)))

# 基于余弦log(snr)的alpha函数
def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

# 学习噪声调度模块
class learned_noise_schedule(nn.Module):
    """ described in section H and then I.2 of the supplementary material for variational ddpm paper """

    def __init__(
        self,
        *,
        log_snr_max,
        log_snr_min,
        hidden_dim = 1024,
        frac_gradient = 1.
    ):
        super().__init__()
        self.slope = log_snr_min - log_snr_max
        self.intercept = log_snr_max

        self.net = nn.Sequential(
            Rearrange('... -> ... 1'),
            MonotonicLinear(1, 1),
            Residual(nn.Sequential(
                MonotonicLinear(1, hidden_dim),
                nn.Sigmoid(),
                MonotonicLinear(hidden_dim, 1)
            )),
            Rearrange('... 1 -> ...'),
        )

        self.frac_gradient = frac_gradient

    def forward(self, x):
        frac_gradient = self.frac_gradient
        device = x.device

        out_zero = self.net(torch.zeros_like(x))
        out_one =  self.net(torch.ones_like(x))

        x = self.net(x)

        normed = self.slope * ((x - out_zero) / (out_one - out_zero)) + self.intercept
        return normed * frac_gradient + normed.detach() * (1 - frac_gradient)

# 连续时间高斯扩散模块
class ContinuousTimeGaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        noise_schedule = 'linear',
        num_sample_steps = 500,
        clip_sample_denoised = True,
        learned_schedule_net_hidden_dim = 1024,
        learned_noise_schedule_frac_gradient = 1.,   # between 0 and 1, determines what percentage of gradients go back, so one can update the learned noise schedule more slowly
        min_snr_loss_weight = False,
        min_snr_gamma = 5
    ):
        # 初始化父类
        super().__init__()
        # 断言模型是否使用随机或学习的正弦条件
        assert model.random_or_learned_sinusoidal_cond
        # 断言模型是否没有自身条件，如果有则抛出异常
        assert not model.self_condition, 'not supported yet'

        self.model = model

        # 图像维度

        self.channels = channels
        self.image_size = image_size

        # 连续噪声计划相关内容

        if noise_schedule == 'linear':
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == 'cosine':
            self.log_snr = alpha_cosine_log_snr
        elif noise_schedule == 'learned':
            # 获取学习的噪声计划的最大和最小值
            log_snr_max, log_snr_min = [beta_linear_log_snr(torch.tensor([time])).item() for time in (0., 1.)]

            self.log_snr = learned_noise_schedule(
                log_snr_max = log_snr_max,
                log_snr_min = log_snr_min,
                hidden_dim = learned_schedule_net_hidden_dim,
                frac_gradient = learned_noise_schedule_frac_gradient
            )
        else:
            # 抛出异常，未知的噪声计划类型
            raise ValueError(f'unknown noise schedule {noise_schedule}')

        # 采样

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised

        # 提议的 https://arxiv.org/abs/2303.09556

        self.min_snr_loss_weight = min_snr_loss_weight
        self.min_snr_gamma = min_snr_gamma

    @property
    def device(self):
        # 返回模型参数的设备
        return next(self.model.parameters()).device

    def p_mean_variance(self, x, time, time_next):
        # 计算均值和方差

        # 根据时间获取对数信噪比
        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])
        pred_noise = self.model(x, batch_log_snr)

        if self.clip_sample_denoised:
            x_start = (x - sigma * pred_noise) / alpha

            # 在 Imagen 中，这被更改为动态阈值
            x_start.clamp_(-1., 1.)

            model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)
        else:
            model_mean = alpha_next / alpha * (x - c * sigma * pred_noise)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance

    # 与采样相关的函数

    @torch.no_grad()
    def p_sample(self, x, time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(x = x, time = time, time_next = time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        batch = shape[0]

        img = torch.randn(shape, device = self.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        img.clamp_(-1., 1.)
        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self, batch_size = 16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # 与训练相关的函数 - 噪声预测

    @autocast(enabled = False)
    # 对输入的起始点 x_start 进行采样，添加噪声，返回添加噪声后的数据和对数信噪比
    def q_sample(self, x_start, times, noise = None):
        # 如果没有提供噪声，则使用默认的噪声生成函数生成噪声
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 计算对数信噪比
        log_snr = self.log_snr(times)

        # 将对数信噪比维度填充到与 x_start 相同的维度
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        # 计算 alpha 和 sigma，用于添加噪声
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        # 添加噪声到 x_start 上
        x_noised =  x_start * alpha + noise * sigma

        # 返回添加噪声后的数据和对数信噪比
        return x_noised, log_snr

    # 生成随机时间点
    def random_times(self, batch_size):
        # 时间点均匀分布在 0 到 1 之间
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    # 计算损失函数
    def p_losses(self, x_start, times, noise = None):
        # 如果没有提供噪声，则使用默认的噪声生成函数生成噪声
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 对起始点 x_start 进行采样，添加噪声，得到模型输出和对数信噪比
        x, log_snr = self.q_sample(x_start = x_start, times = times, noise = noise)
        # 将添加噪声后的数据输入模型，得到模型输出
        model_out = self.model(x, log_snr)

        # 计算均方误差损失
        losses = F.mse_loss(model_out, noise, reduction = 'none')
        # 对损失进行降维处理
        losses = reduce(losses, 'b ... -> b', 'mean')

        # 如果设置了最小信噪比损失权重
        if self.min_snr_loss_weight:
            # 计算信噪比
            snr = log_snr.exp()
            # 计算损失权重
            loss_weight = snr.clamp(min = self.min_snr_gamma) / snr
            # 将损失乘以权重
            losses = losses * loss_weight

        # 返回平均损失
        return losses.mean()

    # 前向传播函数
    def forward(self, img, *args, **kwargs):
        # 获取输入图像的形状和设备信息
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言图像的高度和宽度必须为指定的图像大小
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # 生成随机时间点
        times = self.random_times(b)
        # 将��像归一化到 -1 到 1 之间
        img = normalize_to_neg_one_to_one(img)
        # 计算损失并返回
        return self.p_losses(img, times, *args, **kwargs)
```