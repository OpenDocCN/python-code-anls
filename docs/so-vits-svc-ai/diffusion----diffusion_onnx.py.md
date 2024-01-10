# `so-vits-svc\diffusion\diffusion_onnx.py`

```
# 导入 math 模块
import math
# 从 collections 模块中导入 deque 类
from collections import deque
# 从 functools 模块中导入 partial 函数
from functools import partial
# 从 inspect 模块中导入 isfunction 函数
from inspect import isfunction
# 导入 numpy 模块并重命名为 np
import numpy as np
# 导入 torch 模块
import torch
# 从 torch.nn 模块中导入 functional 模块并重命名为 F
import torch.nn.functional as F
# 从 torch 模块中导入 nn 类
from torch import nn
# 从 torch.nn 模块中导入 Conv1d 类和 Mish 类
from torch.nn import Conv1d, Mish
# 从 tqdm 模块中导入 tqdm 函数
from tqdm import tqdm

# 定义一个函数，判断输入是否存在
def exists(x):
    return x is not None

# 定义一个函数，返回输入值或默认值
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# 定义一个函数，从数组中提取指定位置的元素并重塑形状
def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))

# 定义一个函数，生成指定形状的噪声张量
def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    def noise():
        return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

# 定义一个线性 beta 调度函数
def linear_beta_schedule(timesteps, max_beta=0.02):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas

# 定义一个余弦 beta 调度函数
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

# 定义一个字典，包含不同类型的 beta 调度函数
beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}

# 定义一个函数，从数组中提取指定位置的元素并重塑形状
def extract_1(a, t):
    return a[t].reshape((1, 1, 1, 1))

# 定义一个函数，根据预测噪声和先前噪声预测值计算第 0 阶段的预测值
def predict_stage0(noise_pred, noise_pred_prev):
    return (noise_pred + noise_pred_prev) / 2

# 定义一个函数，根据预测噪声和噪声列表计算第 1 阶段的预测值
def predict_stage1(noise_pred, noise_list):
    return (noise_pred * 3 - noise_list[-1]) / 2

# 定义一个函数，根据预测噪声和噪声列表计算第 2 阶段的预测值
def predict_stage2(noise_pred, noise_list):
    return (noise_pred * 23 - noise_list[-1] * 16 + noise_list[-2] * 5) / 12

# 定义一个函数，根据预测噪声和噪声列表计算第 3 阶段的预测值
def predict_stage3(noise_pred, noise_list):
    # 这里需要继续补充代码和注释
    # 返回噪声预测值的加权平均值
    return (noise_pred * 55
            - noise_list[-1] * 59
            + noise_list[-2] * 37
            - noise_list[-3] * 9) / 24
# 定义一个 SinusoidalPosEmb 类，继承自 nn.Module
class SinusoidalPosEmb(nn.Module):
    # 初始化方法，接受一个参数 dim
    def __init__(self, dim):
        super().__init__()  # 调用父类的初始化方法
        self.dim = dim  # 设置实例变量 dim
        self.half_dim = dim // 2  # 计算 dim 的一半
        self.emb = 9.21034037 / (self.half_dim - 1)  # 计算 emb 的值
        self.emb = torch.exp(torch.arange(self.half_dim) * torch.tensor(-self.emb)).unsqueeze(0)  # 计算 emb 的值
        self.emb = self.emb.cpu()  # 将 emb 转移到 CPU 上

    # 前向传播方法，接受一个参数 x
    def forward(self, x):
        emb = self.emb * x  # 计算 emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # 计算 emb
        return emb  # 返回 emb


# 定义一个 ResidualBlock 类，继承自 nn.Module
class ResidualBlock(nn.Module):
    # 初始化方法，接受三个参数 encoder_hidden, residual_channels, dilation
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()  # 调用父类的初始化方法
        self.residual_channels = residual_channels  # 设置实例变量 residual_channels
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)  # 创建一个 dilated_conv 对象
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)  # 创建一个 diffusion_projection 对象
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)  # 创建一个 conditioner_projection 对象
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)  # 创建一个 output_projection 对象

    # 前向传播方法，接受三个参数 x, conditioner, diffusion_step
    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)  # 计算 diffusion_step
        conditioner = self.conditioner_projection(conditioner)  # 计算 conditioner
        y = x + diffusion_step  # 计算 y
        y = self.dilated_conv(y) + conditioner  # 计算 y

        gate, filter_1 = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)  # 分割 y

        y = torch.sigmoid(gate) * torch.tanh(filter_1)  # 计算 y
        y = self.output_projection(y)  # 计算 y

        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)  # 分割 y

        return (x + residual) / 1.41421356, skip  # 返回结果

# 定义一个 DiffNet 类，继承自 nn.Module
class DiffNet(nn.Module):
    # 初始化函数，设置模型的输入维度、层数、通道数和隐藏层维度
    def __init__(self, in_dims, n_layers, n_chans, n_hidden):
        # 调用父类的初始化函数
        super().__init__()
        # 设置隐藏层维度
        self.encoder_hidden = n_hidden
        # 设置残差层数
        self.residual_layers = n_layers
        # 设置残差通道数
        self.residual_channels = n_chans
        # 创建输入投影层，将输入维度映射到残差通道数维度
        self.input_projection = Conv1d(in_dims, self.residual_channels, 1)
        # 创建扩散嵌入层，用于处理扩散步数
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        # 设置维度为残差通道数
        dim = self.residual_channels
        # 创建多层感知机，用于处理扩散步数
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        # 创建残差层列表，根据残差层数创建多个残差块
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self.encoder_hidden, self.residual_channels, 1)
            for i in range(self.residual_layers)
        ])
        # 创建跳跃投影层，将残差通道数映射回自身
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        # 创建输出投影层，将残差通道数映射回输入维度
        self.output_projection = Conv1d(self.residual_channels, in_dims, 1)
        # 初始化输出投影层的权重为零
        nn.init.zeros_(self.output_projection.weight)

    # 前向传播函数，接收输入spec、扩散步数diffusion_step和条件cond
    def forward(self, spec, diffusion_step, cond):
        # 将输入spec的第一个维度压缩，去除批次维度
        x = spec.squeeze(0)
        # 将压缩后的输入通过输入投影层，映射到残差通道数维度
        x = self.input_projection(x)  # x [B, residual_channel, T]
        # 对映射后的结果进行激活函数处理
        x = F.relu(x)
        # 将扩散步数转换为浮点数类型
        diffusion_step = diffusion_step.float()
        # 通过扩散嵌入层处理扩散步数
        diffusion_step = self.diffusion_embedding(diffusion_step)
        # 通过多层感知机处理扩散步数
        diffusion_step = self.mlp(diffusion_step)

        # 通过第一个残差层处理输入，得到输出和跳跃连接
        x, skip = self.residual_layers[0](x, cond, diffusion_step)
        # 遍历剩余的残差层，依次处理输入，得到输出和跳跃连接
        for layer in self.residual_layers[1:]:
            x, skip_connection = layer.forward(x, cond, diffusion_step)
            skip = skip + skip_connection
        # 对所有跳跃连接求平均，除以残差层数的平方根
        x = skip / math.sqrt(len(self.residual_layers))
        # 通过跳跃投影层处理平均后的跳跃连接
        x = self.skip_projection(x)
        # 对处理后的结果进行激活函数处理
        x = F.relu(x)
        # 通过输出投影层将结果映射回输入维度
        x = self.output_projection(x)  # [B, 80, T]
        # 将结果添加一个维度，表示为批次维度
        return x.unsqueeze(1)
# 定义一个名为 AfterDiffusion 的类，继承自 nn.Module
class AfterDiffusion(nn.Module):
    # 初始化方法，接受 spec_max, spec_min, v_type 三个参数
    def __init__(self, spec_max, spec_min, v_type='a'):
        # 调用父类的初始化方法
        super().__init__()
        # 将参数赋值给对象属性
        self.spec_max = spec_max
        self.spec_min = spec_min
        self.type = v_type

    # 前向传播方法，接受输入 x
    def forward(self, x):
        # 对输入进行压缩和维度置换操作
        x = x.squeeze(1).permute(0, 2, 1)
        # 对输入进行数学运算，得到 mel_out
        mel_out = (x + 1) / 2 * (self.spec_max - self.spec_min) + self.spec_min
        # 如果 v_type 为 'nsf-hifigan-log10'，对 mel_out 进行额外的数学运算
        if self.type == 'nsf-hifigan-log10':
            mel_out = mel_out * 0.434294
        # 将 mel_out 进行维度置换操作后返回
        return mel_out.transpose(2, 1)


# 定义一个名为 Pred 的类，继承自 nn.Module
class Pred(nn.Module):
    # 初始化方法，接受 alphas_cumprod 参数
    def __init__(self, alphas_cumprod):
        # 调用父类的初始化方法
        super().__init__()
        # 将参数赋值给对象属性
        self.alphas_cumprod = alphas_cumprod

    # 前向传播方法，接受输入 x_1, noise_t, t_1, t_prev
    def forward(self, x_1, noise_t, t_1, t_prev):
        # 从 alphas_cumprod 中提取数据，进行数学运算得到 x_pred
        a_t = extract(self.alphas_cumprod, t_1).cpu()
        a_prev = extract(self.alphas_cumprod, t_prev).cpu()
        a_t_sq, a_prev_sq = a_t.sqrt().cpu(), a_prev.sqrt().cpu()
        x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x_1 - 1 / (
                a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x_1 + x_delta.cpu()
        # 返回 x_pred
        return x_pred


# 定义一个名为 GaussianDiffusion 的类，继承自 nn.Module
class GaussianDiffusion(nn.Module):
    # 定义 q_mean_variance 方法，接受 x_start, t 两个参数
    def q_mean_variance(self, x_start, t):
        # 计算均值、方差和对数方差
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        # 返回均值、方差和对数方差
        return mean, variance, log_variance

    # 定义 predict_start_from_noise 方法，接受 x_t, t, noise 三个参数
    def predict_start_from_noise(self, x_t, t, noise):
        # 根据输入进行数学运算，返回结果
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    # 计算后验分布的均值、方差和截断后的对数方差
    def q_posterior(self, x_start, x_t, t):
        # 计算后验分布的均值
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # 提取后验分布的方差
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        # 提取截断后的对数方差
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 计算模型的均值、方差和对数方差
    def p_mean_variance(self, x, t, cond):
        # 使用去噪函数对输入进行去噪
        noise_pred = self.denoise_fn(x, t, cond=cond)
        # 从去噪后的输入预测起始状态
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        # 将重构后的输入限制在[-1, 1]范围内
        x_recon.clamp_(-1., 1.)

        # 计算后验分布的均值、方差和对数方差
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # 从模型中采样
    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False):
        # 获取输入的维度和设备信息
        b, *_, device = *x.shape, x.device
        # 计算模型的均值、方差和对数方差
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        # 生成与输入相同维度的噪声
        noise = noise_like(x.shape, device, repeat_noise)
        # 当 t == 0 时不添加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 禁用梯度计算
    @torch.no_grad()
    # 使用 PLMS 方法从给定链接中提取的伪数值方法进行采样
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from
        [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        # 获取预测的 x 值
        def get_x_pred(x, noise_t, t):
            # 提取 t 时刻的 alpha 累积乘积
            a_t = extract(self.alphas_cumprod, t)
            # 提取 t 时刻前 interval 时刻的 alpha 累积乘积
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)))
            # 对 alpha 累积乘积进行平方根处理
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            # 计算 x 的变化量
            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                    a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            # 计算预测的 x 值
            x_pred = x + x_delta

            return x_pred

        # 获取噪声列表和预测的噪声
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

        # 获取前一个 x 值
        x_prev = get_x_pred(x, noise_pred_prime, t)
        # 将预测的噪声添加到噪声列表中
        noise_list.append(noise_pred)

        return x_prev

    # 对给定的 x_start 和 t 进行采样
    def q_sample(self, x_start, t, noise=None):
        # 如果没有提供噪声，则使用标准正态分布生成噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        # 返回采样结果
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    # 计算损失函数
    def p_losses(self, x_start, t, cond, noise=None, loss_type='l2'):
        # 如果没有提供噪声，则生成一个与 x_start 相同形状的随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # 使用噪声生成含噪声的样本
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 使用去噪函数对含噪声的样本进行去噪
        x_recon = self.denoise_fn(x_noisy, t, cond)

        # 根据损失类型计算损失
        if loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif loss_type == 'l2':
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

    # 根据给定的输入和时间步长，以及之前的时间步长，预测下一个时间步长的输入
    def get_x_pred(self, x_1, noise_t, t_1, t_prev):
        # 提取当前时间步长和之前时间步长对应的累积因子
        a_t = extract(self.alphas_cumprod, t_1)
        a_prev = extract(self.alphas_cumprod, t_prev)
        # 对累积因子进行平方根处理
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()
        # 计算输入的变化量
        x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x_1 - 1 / (
                a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
        # 根据输入的变化量计算预测的输入
        x_pred = x_1 + x_delta
        # 返回预测的输入
        return x_pred
    # 定义一个方法，用于执行前向传播
    def forward(self, condition=None, init_noise=None, pndms=None, k_step=None):
        # 将传入的条件赋值给变量cond
        cond = condition
        # 将初始噪声赋值给变量x
        x = init_noise

        # 获取条件的设备信息
        device = cond.device
        # 获取条件的帧数
        n_frames = cond.shape[2]
        # 创建一个步长范围的张量
        step_range = torch.arange(0, k_step.item(), pndms.item(), dtype=torch.long, device=device).flip(0)
        # 创建一个长整型的张量plms_noise_stage
        plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
        # 创建一个全零张量noise_list
        noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)

        # 遍历步长范围
        for t in step_range:
            # 创建一个张量t_1，值为t
            t_1 = torch.full((1,), t, device=device, dtype=torch.long)
            # 使用噪声去噪函数对噪声进行预测
            noise_pred = self.denoise_fn(x, t_1, cond)
            # 计算t_prev
            t_prev = t_1 - pndms
            t_prev = t_prev * (t_prev > 0)
            # 根据plms_noise_stage的值选择不同的预测函数
            if plms_noise_stage == 0:
                x_pred = self.get_x_pred(x, noise_pred, t_1, t_prev)
                noise_pred_prev = self.denoise_fn(x_pred, t_prev, cond=cond)
                noise_pred_prime = predict_stage0(noise_pred, noise_pred_prev)

            elif plms_noise_stage == 1:
                noise_pred_prime = predict_stage1(noise_pred, noise_list)

            elif plms_noise_stage == 2:
                noise_pred_prime = predict_stage2(noise_pred, noise_list)

            else:
                noise_pred_prime = predict_stage3(noise_pred, noise_list)

            # 在维度0上增加一个维度
            noise_pred = noise_pred.unsqueeze(0)

            # 根据plms_noise_stage的值进行不同的操作
            if plms_noise_stage < 3:
                noise_list = torch.cat((noise_list, noise_pred), dim=0)
                plms_noise_stage = plms_noise_stage + 1

            else:
                noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)

            # 更新变量x
            x = self.get_x_pred(x, noise_pred_prime, t_1, t_prev)
        # 对x进行自适应处理
        x = self.ad(x)
        # 返回处理后的x
        return x
```