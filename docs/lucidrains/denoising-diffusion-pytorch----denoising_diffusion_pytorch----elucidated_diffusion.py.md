# `.\lucidrains\denoising-diffusion-pytorch\denoising_diffusion_pytorch\elucidated_diffusion.py`

```py
# 从 math 模块中导入 sqrt 函数
from math import sqrt
# 从 random 模块中导入 random 函数
from random import random
# 导入 torch 模块
import torch
# 从 torch 模块中导入 nn, einsum 函数
from torch import nn, einsum
# 从 torch.nn 模块中导入 functional 模块
import torch.nn.functional as F
# 从 tqdm 模块中导入 tqdm 函数
from tqdm import tqdm
# 从 einops 模块中导入 rearrange, repeat, reduce 函数

# helpers

# 定义函数 exists，判断值是否存在
def exists(val):
    return val is not None

# 定义函数 default，返回默认值
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# tensor helpers

# 定义函数 log，计算张量的对数
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

# normalization functions

# 定义函数 normalize_to_neg_one_to_one，将图像归一化到 [-1, 1] 范围内
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

# 定义函数 unnormalize_to_zero_to_one，将张量反归一化到 [0, 1] 范围内
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# main class

# 定义类 ElucidatedDiffusion，继承自 nn.Module 类
class ElucidatedDiffusion(nn.Module):
    # 初始化方法
    def __init__(
        self,
        net,
        *,
        image_size,
        channels = 3,
        num_sample_steps = 32, # number of sampling steps
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 0.5,      # standard deviation of data distribution
        rho = 7,               # controls the sampling schedule
        P_mean = -1.2,         # mean of log-normal distribution from which noise is drawn for training
        P_std = 1.2,           # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn = 80,          # parameters for stochastic sampling - depends on dataset, Table 5 in apper
        S_tmin = 0.05,
        S_tmax = 50,
        S_noise = 1.003,
    ):
        super().__init__()
        assert net.random_or_learned_sinusoidal_cond
        self.self_condition = net.self_condition

        self.net = net

        # image dimensions

        self.channels = channels
        self.image_size = image_size

        # parameters

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.rho = rho

        self.P_mean = P_mean
        self.P_std = P_std

        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper

        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    # 获取设备信息
    @property
    def device(self):
        return next(self.net.parameters()).device

    # derived preconditioning params - Table 1

    # 计算 c_skip 参数
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    # 计算 c_out 参数
    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    # 计算 c_in 参数
    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    # 计算 c_noise 参数
    def c_noise(self, sigma):
        return log(sigma) * 0.25

    # preconditioned network output
    # equation (7) in the paper

    # 预处理网络输出
    def preconditioned_network_forward(self, noised_images, sigma, self_cond = None, clamp = False):
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device = device)

        padded_sigma = rearrange(sigma, 'b -> b 1 1 1')

        net_out = self.net(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            self_cond
        )

        out = self.c_skip(padded_sigma) * noised_images +  self.c_out(padded_sigma) * net_out

        if clamp:
            out = out.clamp(-1., 1.)

        return out

    # sampling

    # sample schedule
    # equation (5) in the paper

    # 采样计划
    def sample_schedule(self, num_sample_steps = None):
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device = self.device, dtype = torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (N - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value = 0.) # last step is sigma value of 0.
        return sigmas

    @torch.no_grad()
    # 定义一个方法用于生成样本，可以设置批量大小、采样步数和是否进行截断
    def sample(self, batch_size = 16, num_sample_steps = None, clamp = True):
        # 如果未指定采样步数，则使用默认值
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        # 定义图像形状
        shape = (batch_size, self.channels, self.image_size, self.image_size)

        # 获取采样计划，返回(sigma, gamma)元组，并与下一个sigma和gamma配对
        sigmas = self.sample_schedule(num_sample_steps)

        # 计算gamma值
        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1])

        # 初始化图像为噪声
        init_sigma = sigmas[0]
        images = init_sigma * torch.randn(shape, device = self.device)

        # 用于自我条件
        x_start = None

        # 逐步去噪
        for sigma, sigma_next, gamma in tqdm(sigmas_and_gammas, desc = 'sampling time step'):
            sigma, sigma_next, gamma = map(lambda t: t.item(), (sigma, sigma_next, gamma))

            # 随机采样
            eps = self.S_noise * torch.randn(shape, device = self.device)

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat ** 2 - sigma ** 2) * eps

            self_cond = x_start if self.self_condition else None

            model_output = self.preconditioned_network_forward(images_hat, sigma_hat, self_cond, clamp = clamp)
            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            # 如果不是最后一个时间步，进行二阶修正
            if sigma_next != 0:
                self_cond = model_output if self.self_condition else None

                model_output_next = self.preconditioned_network_forward(images_next, sigma_next, self_cond, clamp = clamp)
                denoised_prime_over_sigma = (images_next - model_output_next) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (denoised_over_sigma + denoised_prime_over_sigma)

            images = images_next
            x_start = model_output_next if sigma_next != 0 else model_output

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    @torch.no_grad()
    # 使用DPMPP进行采样
    def sample_using_dpmpp(self, batch_size = 16, num_sample_steps = None):
        """
        感谢Katherine Crowson (https://github.com/crowsonkb)解决了所有问题！
        https://arxiv.org/abs/2211.01095
        """

        device, num_sample_steps = self.device, default(num_sample_steps, self.num_sample_steps)

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        images  = sigmas[0] * torch.randn(shape, device = device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(images, sigmas[i].item())
            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = - 1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (-h).expm1() * denoised_d
            old_denoised = denoised

        images = images.clamp(-1., 1.)
        return unnormalize_to_zero_to_one(images)

    # 计算损失权重
    def loss_weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) * (sigma * self.sigma_data) ** -2
    # 计算噪声分布，返回一个服从指定均值和标准差的正态分布的随机数张量
    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device = self.device)).exp()

    # 前向传播函数
    def forward(self, images):
        # 获取输入图片的形状信息
        batch_size, c, h, w, device, image_size, channels = *images.shape, images.device, self.image_size, self.channels

        # 断言输入图片的高度和宽度与指定的图像大小相同
        assert h == image_size and w == image_size, f'height and width of image must be {image_size}'
        # 断言输入图片的通道数与指定的通道数相同
        assert c == channels, 'mismatch of image channels'

        # 将输入图片归一化到[-1, 1]范围内
        images = normalize_to_neg_one_to_one(images)

        # 生成噪声标准差
        sigmas = self.noise_distribution(batch_size)
        # 将噪声标准差扩展为与输入图片相同的形状
        padded_sigmas = rearrange(sigmas, 'b -> b 1 1 1')

        # 生成与输入图片相同形状的随机噪声
        noise = torch.randn_like(images)

        # 对输入图片添加噪声，噪声系数为噪声标准差
        noised_images = images + padded_sigmas * noise  # alphas are 1. in the paper

        # 初始化自条件变量
        self_cond = None

        # 如果开启了自条件功能且随机数小于0.5
        if self.self_condition and random() < 0.5:
            # 从Hinton小组的位扩散论文中获取
            with torch.no_grad():
                # 计算预处理网络的前向传播结果
                self_cond = self.preconditioned_network_forward(noised_images, sigmas)
                # 分离自条件变量
                self_cond.detach_()

        # 对添加噪声后的图片进行去噪处理
        denoised = self.preconditioned_network_forward(noised_images, sigmas, self_cond)

        # 计算去噪损失，使用均方误差损失函数
        losses = F.mse_loss(denoised, images, reduction = 'none')
        # 对损失进行降维处理
        losses = reduce(losses, 'b ... -> b', 'mean')

        # 根据噪声标准差调整损失权重
        losses = losses * self.loss_weight(sigmas)

        # 返回平均损失值
        return losses.mean()
```