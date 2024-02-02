# `so-vits-svc\diffusion\diffusion.py`

```py
# 导入所需的库
from collections import deque
from functools import partial
from inspect import isfunction
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# 定义函数，判断变量是否存在
def exists(x):
    return x is not None

# 定义函数，返回val或者默认值d
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 定义函数，从张量a中提取指定索引t对应的值，并reshape成指定形状
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 定义函数，生成指定形状的噪声张量
def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    def noise():
        return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

# 定义函数，生成线性beta调度表
def linear_beta_schedule(timesteps, max_beta=0.02):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas

# 定义函数，生成余弦beta调度表
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)

# 定义字典，包含不同类型的beta调度生成函数
beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}

# 定义类，高斯扩散
class GaussianDiffusion(nn.Module):
    # 定义函数，计算q分布的均值、方差和对数方差
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    # 根据给定的输入和噪声，预测起始值
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                # 从累积平方根倒数系数中提取对应时刻的值，与输入相乘
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                # 从累积平方根倒数减一系数中提取对应时刻的值，与噪声相乘
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # 计算后验分布的均值、方差和截断后的对数方差
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                # 从后验均值系数1中提取对应时刻的值，与起始值相乘
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                # 从后验均值系数2中提取对应时刻的值，与输入相乘
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)  # 提取后验方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)  # 提取截断后的对数方差
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 计算模型的均值、方差和对数方差
    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond=cond)  # 使用去噪函数得到噪声预测
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)  # 使用预测起始值函数得到重构值

        x_recon.clamp_(-1., 1.)  # 对重构值进行截断，限制在[-1, 1]范围内

        # 计算后验分布的均值、方差和截断后的对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    # 使用 DDIM 方法进行多维采样
    def p_sample_ddim(self, x, t, interval, cond):
        """
        Use the DDIM method from
        """
        a_t = extract(self.alphas_cumprod, t, x.shape)  # 从累积乘积系数中提取对应时刻的值
        a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)  # 从累积乘积系数中提取前一个时刻的值

        noise_pred = self.denoise_fn(x, t, cond=cond)  # 使用去噪函数得到噪声预测
        x_prev = a_prev.sqrt() * (x / a_t.sqrt() + (((1 - a_prev) / a_prev).sqrt()-((1 - a_t) / a_t).sqrt()) * noise_pred)  # 计算前一个时刻的值
        return x_prev
    # 定义一个函数，用于生成样本
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        # 获取输入张量 x 的形状，并获取设备信息
        b, *_, device = *x.shape, x.device
        # 调用 p_mean_variance 函数，获取模型的均值和对数方差
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        # 生成与输入张量 x 相同形状的噪声
        noise = noise_like(x.shape, device, repeat_noise)
        # 当 t == 0 时，不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        # 返回生成的样本，根据模型均值、对数方差和噪声计算得到
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 禁用梯度计算
    @torch.no_grad()
    # 使用 PLMS 方法对样本进行处理
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        使用[Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778)中的PLMS方法。
        """

        # 获取预测的x值
        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)), x.shape)
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                    a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond)

        # 根据噪声列表的长度进行不同的处理
        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond=cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    # 对样本进行采样
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # 计算生成器的损失函数
    def p_losses(self, x_start, t, cond, noise=None, loss_type='l2'):
        # 如果没有提供噪声，则生成一个与 x_start 相同形状的随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # 使用生成器对输入进行采样，得到带噪声的输出
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 使用去噪函数对带噪声的输出进行去噪，得到重构的输出
        x_recon = self.denoise_fn(x_noisy, t, cond)

        # 根据损失类型计算损失
        if loss_type == 'l1':
            # 使用 L1 损失函数计算损失
            loss = (noise - x_recon).abs().mean()
        elif loss_type == 'l2':
            # 使用均方误差损失函数计算损失
            loss = F.mse_loss(noise, x_recon)
        else:
            # 如果损失类型不是 l1 或 l2，则抛出未实现的错误
            raise NotImplementedError()

        # 返回计算得到的损失
        return loss

    # 对输入进行规范化
    def norm_spec(self, x):
        return (x - self.spec_min) / (self.spec_max - self.spec_min) * 2 - 1

    # 对规范化后的输入进行反规范化
    def denorm_spec(self, x):
        return (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
```