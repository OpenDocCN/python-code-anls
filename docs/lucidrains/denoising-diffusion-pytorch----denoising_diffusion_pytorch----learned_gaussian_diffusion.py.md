# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\learned_gaussian_diffusion.py`

```py
import torch
from collections import namedtuple
from math import pi, sqrt, log as ln
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion, extract, unnormalize_to_zero_to_one

# 定义常量
NAT = 1. / ln(2)

# 定义命名元组
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_variance'])

# 辅助函数

# 判断变量是否存在
def exists(x):
    return x is not None

# 返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 张量辅助函数

# 计算张量的对数
def log(t, eps = 1e-15):
    return torch.log(t.clamp(min = eps))

# 求张量的平均值
def meanflat(x):
    return x.mean(dim = tuple(range(1, len(x.shape)))

# 计算两个正态分布之间的 KL 散度
def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

# 近似标准正态分布的累积分布函数
def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * (x ** 3)))

# 计算离散高斯分布的对数似然
def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres = 0.999):
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1. - cdf_min)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(x < -thres,
        log_cdf_plus,
        torch.where(x > thres,
            log_one_minus_cdf_min,
            log(cdf_delta)))

    return log_probs

# https://arxiv.org/abs/2102.09672

# i thought the results were questionable, if one were to focus only on FID
# but may as well get this in here for others to try, as GLIDE is using it (and DALL-E2 first stage of cascade)
# gaussian diffusion for learned variance + hybrid eps simple + vb loss

# 继承 GaussianDiffusion 类，实现 LearnedGaussianDiffusion 类
class LearnedGaussianDiffusion(GaussianDiffusion):
    def __init__(
        self,
        model,
        vb_loss_weight = 0.001,  # lambda was 0.001 in the paper
        *args,
        **kwargs
    ):
        super().__init__(model, *args, **kwargs)
        assert model.out_dim == (model.channels * 2), 'dimension out of unet must be twice the number of channels for learned variance - you can also set the `learned_variance` keyword argument on the Unet to be `True`'
        assert not model.self_condition, 'not supported yet'

        self.vb_loss_weight = vb_loss_weight

    # 模型预测函数
    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, rederive_pred_noise = False):
        model_output = self.model(x, t)
        model_output, pred_variance = model_output.chunk(2, dim = 1)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        x_start = maybe_clip(x_start)

        return ModelPrediction(pred_noise, x_start, pred_variance)
    # 计算预测均值、方差和对数方差，根据输入的特征 x 和时间 t，以及是否裁剪去噪声
    def p_mean_variance(self, *, x, t, clip_denoised, model_output = None, **kwargs):
        # 如果未提供模型输出，则使用默认的模型输出函数计算模型输出
        model_output = default(model_output, lambda: self.model(x, t))
        # 将模型输出分成预测噪声和插值分数未归一化的方差
        pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim = 1)

        # 提取后验对数方差的最小值和最大值
        min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
        max_log = extract(torch.log(self.betas), t, x.shape)
        # 将插值分数未归一化的方差归一化到 [0, 1] 区间
        var_interp_frac = unnormalize_to_zero_to_one(var_interp_frac_unnormalized)

        # 计算模型对数方差和方差
        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        model_variance = model_log_variance.exp()

        # 根据预测噪声和时间 t 预测起始值 x_start
        x_start = self.predict_start_from_noise(x, t, pred_noise)

        # 如果需要裁剪去噪声，则将 x_start 裁剪到 [-1, 1] 区间
        if clip_denoised:
            x_start.clamp_(-1., 1.)

        # 计算模型均值和其他参数
        model_mean, _, _ = self.q_posterior(x_start, x, t)

        # 返回模型均值、方差、对数方差和起始值 x_start
        return model_mean, model_variance, model_log_variance, x_start

    # 计算损失函数，包括 KL 散度和简单损失
    def p_losses(self, x_start, t, noise = None, clip_denoised = False):
        # 如果未提供噪声，则使用默认的噪声函数生成噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 根据起始值 x_start、时间 t 和噪声生成 x_t
        x_t = self.q_sample(x_start = x_start, t = t, noise = noise)

        # 获取模型输出
        model_output = self.model(x_t, t)

        # 计算学习方差（插值）的 KL 散度
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start = x_start, x_t = x_t, t = t)
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x = x_t, t = t, clip_denoised = clip_denoised, model_output = model_output)

        # 为了稳定性，使用分离的模型预测均值计算 KL 散度
        detached_model_mean = model_mean.detach()

        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT

        # 计算解码器负对数似然
        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means = detached_model_mean, log_scales = 0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT

        # 在第一个时间步返回解码器 NLL，否则返回 KL 散度
        vb_losses = torch.where(t == 0, decoder_nll, kl)

        # 简单损失 - 预测噪声、x0 或 x_prev
        pred_noise, _ = model_output.chunk(2, dim = 1)
        simple_losses = F.mse_loss(pred_noise, noise)

        # 返回简单损失和 VB 损失的平均值乘以 VB 损失权重
        return simple_losses + vb_losses.mean() * self.vb_loss_weight
```