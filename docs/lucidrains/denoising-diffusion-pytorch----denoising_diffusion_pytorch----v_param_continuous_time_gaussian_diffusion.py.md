# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\v_param_continuous_time_gaussian_diffusion.py`

```py
# 导入数学库和PyTorch库
import math
import torch
# 从torch库中导入sqrt函数
from torch import sqrt
# 从torch库中导入nn、einsum模块
from torch import nn, einsum
# 从torch库中导入F模块
import torch.nn.functional as F
# 从torch.special库中导入expm1函数
from torch.special import expm1
# 从torch.cuda.amp库中导入autocast函数

# 从tqdm库中导入tqdm函数
from tqdm import tqdm
# 从einops库中导入rearrange、repeat、reduce函数
from einops import rearrange, repeat, reduce
# 从einops.layers.torch库中导入Rearrange类

# helpers

# 判断变量是否存在的函数
def exists(val):
    return val is not None

# 如果变量存在则返回该变量，否则返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# normalization functions

# 将图像归一化到[-1, 1]范围内
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 将张量反归一化到[0, 1]范围内
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# diffusion helpers

# 将t张量的维度右侧填充到与x张量相同维度
def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# continuous schedules
# log(snr) that approximates the original linear schedule

# 计算t的对数，避免t小于eps时取对数出错
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# 计算alpha_cosine_log_snr函数
def alpha_cosine_log_snr(t, s = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5)

# 定义VParamContinuousTimeGaussianDiffusion类
class VParamContinuousTimeGaussianDiffusion(nn.Module):
    """
    a new type of parameterization in v-space proposed in https://arxiv.org/abs/2202.00512 that
    (1) allows for improved distillation over noise prediction objective and
    (2) noted in imagen-video to improve upsampling unets by removing the color shifting artifacts
    """

    # 初始化函数
    def __init__(
        self,
        model,
        *,
        image_size,
        channels = 3,
        num_sample_steps = 500,
        clip_sample_denoised = True,
    ):
        super().__init__()
        assert model.random_or_learned_sinusoidal_cond
        assert not model.self_condition, 'not supported yet'

        self.model = model

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # continuous noise schedule related stuff

        self.log_snr = alpha_cosine_log_snr

        # sampling

        self.num_sample_steps = num_sample_steps
        self.clip_sample_denoised = clip_sample_denoised        

    # 获取设备信息
    @property
    def device(self):
        return next(self.model.parameters()).device

    # 计算p_mean_variance函数
    def p_mean_variance(self, x, time, time_next):
        # reviewer found an error in the equation in the paper (missing sigma)
        # following - https://openreview.net/forum?id=2LdBqxc1Yv&noteId=rIQgH0zKsRt

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b = x.shape[0])

        pred_v = self.model(x, batch_log_snr)

        # shown in Appendix D in the paper
        x_start = alpha * x - sigma * pred_v

        if self.clip_sample_denoised:
            x_start.clamp_(-1., 1.)

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

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
    # 定义一个函数，用于执行采样循环
    def p_sample_loop(self, shape):
        # 获取批次大小
        batch = shape[0]

        # 生成指定形状的随机张量
        img = torch.randn(shape, device = self.device)
        # 在指定范围内生成一系列步长
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device = self.device)

        # 循环执行采样步骤
        for i in tqdm(range(self.num_sample_steps), desc = 'sampling loop time step', total = self.num_sample_steps):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(img, times, times_next)

        # 将张量值限制在指定范围内
        img.clamp_(-1., 1.)
        # 将张量值从[-1, 1]范围转换为[0, 1]范围
        img = unnormalize_to_zero_to_one(img)
        return img

    # 无梯度计算的采样函数
    @torch.no_grad()
    def sample(self, batch_size = 16):
        return self.p_sample_loop((batch_size, self.channels, self.image_size, self.image_size))

    # 训练相关函数 - 噪声预测

    # 生成采样数据
    @autocast(enabled = False)
    def q_sample(self, x_start, times, noise = None):
        # 生成随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 计算信噪比
        log_snr = self.log_snr(times)

        # 对信噪比进行填充
        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr, alpha, sigma

    # 生成随机时间
    def random_times(self, batch_size):
        return torch.zeros((batch_size,), device = self.device).float().uniform_(0, 1)

    # 计算损失函数
    def p_losses(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x, log_snr, alpha, sigma = self.q_sample(x_start = x_start, times = times, noise = noise)

        # 描述在第4节中作为预测目标，附录D中有推导
        v = alpha * noise - sigma * x_start

        model_out = self.model(x, log_snr)

        return F.mse_loss(model_out, v)

    # 前向传播函数
    def forward(self, img, *args, **kwargs):
        # 获取输入图像的形状和设备信息
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # 断言图像的高度和宽度必须为指定大小
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # 生成随机时间
        times = self.random_times(b)
        # 将图像值从[-1, 1]范围转换���[-1, 1]范围
        img = normalize_to_neg_one_to_one(img)
        return self.p_losses(img, times, *args, **kwargs)
```