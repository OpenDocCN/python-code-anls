# `.\lucidrains\ddpm-ipa-protein-generation\ddpm_ipa_protein_generation\ddpm_ipa_protein_generation.py`

```
import torch
from torch import nn

# gaussian diffusion with continuous time helper functions and classes
# large part of this was thanks to @crowsonkb at https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/utils.py

# 定义一个 torch.jit.script 装饰器修饰的函数，用于计算 beta_linear_log_snr
@torch.jit.script
def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

# 定义一个 torch.jit.script 装饰器修饰的函数，用于计算 alpha_cosine_log_snr
@torch.jit.script
def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version

# 将 log_snr 转换为 alpha 和 sigma
def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

# 定义一个继承自 nn.Module 的类 Diffusion
class Diffusion(nn.Module):
    def __init__(self, *, noise_schedule, timesteps = 1000):
        super().__init__()

        # 根据噪声调度选择不同的 log_snr 函数
        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.num_timesteps = timesteps

    # 获取时间
    def get_times(self, batch_size, noise_level, *, device):
        return torch.full((batch_size,), noise_level, device = device, dtype = torch.float32)

    # 随机采样时间
    def sample_random_times(self, batch_size, max_thres = 0.999, *, device):
        return torch.zeros((batch_size,), device = device).float().uniform_(0, max_thres)

    # 获取条件
    def get_condition(self, times):
        return maybe(self.log_snr)(times)

    # 获取采样时间步长
    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.num_timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    # 计算 posterior
    def q_posterior(self, x_start, x_t, t, *, t_next = None):
        t_next = default(t_next, lambda: (t - 1. / self.num_timesteps).clamp(min = 0.))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.log_snr(t)
        log_snr_next = self.log_snr(t_next)
        log_snr, log_snr_next = map(partial(right_pad_dims_to, x_t), (log_snr, log_snr_next))

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next ** 2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps = 1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 采样
    def q_sample(self, x_start, t, noise = None):
        if isinstance(t, float):
            batch = x_start.shape[0]
            t = torch.full((batch,), t, device = x_start.device, dtype = x_start.dtype)

        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.log_snr(t)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)
        return alpha * x_start + sigma * noise, log_snr
    # 从输入的起始点 x_from 开始，根据给定的时间范围 from_t 到 to_t，生成样本
    def q_sample_from_to(self, x_from, from_t, to_t, noise = None):
        # 获取输入张量 x_from 的形状、设备和数据类型
        shape, device, dtype = x_from.shape, x_from.device, x_from.dtype
        batch = shape[0]

        # 如果 from_t 是浮点数，则将其转换为与 batch 大小相同的张量
        if isinstance(from_t, float):
            from_t = torch.full((batch,), from_t, device = device, dtype = dtype)

        # 如果 to_t 是浮点数，则将其转换为与 batch 大小相同的张量
        if isinstance(to_t, float):
            to_t = torch.full((batch,), to_t, device = device, dtype = dtype)

        # 如果未提供噪声，则生成一个与 x_from 相同形状的随机噪声张量
        noise = default(noise, lambda: torch.randn_like(x_from))

        # 计算起始点到终点的 log_snr，并在需要时对其进行维度填充
        log_snr = self.log_snr(from_t)
        log_snr_padded_dim = right_pad_dims_to(x_from, log_snr)
        alpha, sigma =  log_snr_to_alpha_sigma(log_snr_padded_dim)

        # 计算终点的 log_snr，并在需要时对其进行维度填充
        log_snr_to = self.log_snr(from_t)
        log_snr_padded_dim_to = right_pad_dims_to(x_from, log_snr_to)
        alpha_to, sigma_to =  log_snr_to_alpha_sigma(log_snr_padded_dim_to)

        # 根据公式生成样本并返回
        return x_from * (alpha_to / alpha) + noise * (sigma_to * alpha - sigma * alpha_to) / alpha

    # 根据给定的时间 t 和噪声生成预测的起始点
    def predict_start_from_noise(self, x_t, t, noise):
        # 计算时间 t 对应的 log_snr，并在需要时对其进行维度填充
        log_snr = self.log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        
        # 根据公式计算并返回预测的起始点
        return (x_t - sigma * noise) / alpha.clamp(min = 1e-8)
```