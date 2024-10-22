# `.\diffusers\schedulers\scheduling_ddim.py`

```py
# 版权声明，说明文件的版权所有者及其许可信息
# Copyright 2024 Stanford University Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可协议授权使用
# Licensed under the Apache License, Version 2.0 (the "License");
# 只能在遵守许可的情况下使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下地址获取许可的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面协议另有约定，否则软件在“按原样”基础上分发，不附带任何明示或暗示的保证
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定语言的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 声明：此代码深受以下项目的影响
# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

# 导入数学库
import math
# 从数据类模块导入数据类装饰器
from dataclasses import dataclass
# 导入类型注解相关的类型
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch

# 从配置工具中导入配置混合类和注册配置函数
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具中导入基础输出类
from ..utils import BaseOutput
# 从 PyTorch 工具中导入生成随机张量的函数
from ..utils.torch_utils import randn_tensor
# 从调度工具中导入 Karras 扩散调度器和调度混合类
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
# 从 diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput 复制的调度器输出类，修改了 DDPM 为 DDIM
class DDIMSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像):
            上一时间步的计算样本 `(x_{t-1})`。`prev_sample` 应作为下一次模型输入
            在去噪循环中使用。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像):
            基于当前时间步模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或进行引导。
    """

    # 定义前一个样本张量
    prev_sample: torch.Tensor
    # 定义可选的预测原始样本张量
    pred_original_sample: Optional[torch.Tensor] = None


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建一个 beta 调度，离散化给定的 alpha_t_bar 函数，该函数定义了随时间变化的
    (1-beta) 的累积乘积，t 的范围为 [0,1]。

    包含一个 alpha_bar 函数，该函数接受 t 参数并将其转换为
    (1-beta) 的累积乘积，直到扩散过程的该部分。

    参数：
        num_diffusion_timesteps (`int`): 要生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以
                     防止奇异情况。
        alpha_transform_type (`str`, *可选*，默认为 `cosine`): alpha_bar 的噪声调度类型。
                     可选值为 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于逐步模型输出的 beta 值
    """
    # 检查给定的 alpha 变换类型是否为 "cosine"
        if alpha_transform_type == "cosine":
    
            # 定义一个函数，根据输入 t 计算 cos 变换
            def alpha_bar_fn(t):
                # 返回经过调整的 cos 函数值的平方
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    
        # 检查给定的 alpha 变换类型是否为 "exp"
        elif alpha_transform_type == "exp":
    
            # 定义一个函数，根据输入 t 计算指数变换
            def alpha_bar_fn(t):
                # 返回 e 的 t 值乘以 -12 的指数
                return math.exp(t * -12.0)
    
        # 如果 alpha 变换类型不支持，则抛出异常
        else:
            raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    
        # 初始化一个空列表用于存储 beta 值
        betas = []
        # 遍历每一个扩散时间步
        for i in range(num_diffusion_timesteps):
            # 计算当前时间步的比例 t1
            t1 = i / num_diffusion_timesteps
            # 计算下一个时间步的比例 t2
            t2 = (i + 1) / num_diffusion_timesteps
            # 计算并追加 beta 值到列表，确保不超过 max_beta
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        # 返回一个包含 beta 值的张量，数据类型为 float32
        return torch.tensor(betas, dtype=torch.float32)
# 定义一个函数，重新缩放 betas 以使终端 SNR 为零
def rescale_zero_terminal_snr(betas):
    """
    根据 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 重新缩放 betas 以实现零终端 SNR

    参数:
        betas (`torch.Tensor`):
            初始化调度器时使用的 betas。

    返回:
        `torch.Tensor`: 具有零终端 SNR 的重新缩放的 betas
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算 alphas 的平方根

    # 存储旧值
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 克隆第一个值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 克隆最后一个值

    # 平移使得最后一步为零
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 减去最后一个值

    # 缩放使得第一步恢复到旧值
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 缩放

    # 将 alphas_bar_sqrt 转换回 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 还原累积乘积
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个元素拼接回来
    betas = 1 - alphas  # 计算新的 betas

    return betas  # 返回重新缩放的 betas


# 定义 DDIMScheduler 类，扩展去噪程序
class DDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDIMScheduler` 扩展了在去噪扩散概率模型 (DDPMs) 中引入的去噪程序，
    并使用非马尔可夫引导。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查看超类文档以获取
    所有调度器的通用方法，例如加载和保存。

    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]  # 兼容的调度器列表
    order = 1  # 设置调度器的顺序

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,  # 训练时间步数
        beta_start: float = 0.0001,  # beta 的起始值
        beta_end: float = 0.02,  # beta 的结束值
        beta_schedule: str = "linear",  # beta 调度方式
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 已训练的 betas
        clip_sample: bool = True,  # 是否剪切样本
        set_alpha_to_one: bool = True,  # 是否将 alpha 设置为 1
        steps_offset: int = 0,  # 步骤偏移量
        prediction_type: str = "epsilon",  # 预测类型
        thresholding: bool = False,  # 是否进行阈值处理
        dynamic_thresholding_ratio: float = 0.995,  # 动态阈值比例
        clip_sample_range: float = 1.0,  # 样本范围
        sample_max_value: float = 1.0,  # 样本最大值
        timestep_spacing: str = "leading",  # 时间步间隔方式
        rescale_betas_zero_snr: bool = False,  # 是否重新缩放以实现零 SNR
    ):
        # 如果训练的 beta 值不为 None，则将其转换为张量
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果 beta 调度是线性的，生成一个线性间隔的 beta 值张量
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果 beta 调度是 scaled_linear，特定于潜在扩散模型，生成平方根后再平方的 beta 值
        elif beta_schedule == "scaled_linear":
            # 该调度非常特定于潜在扩散模型
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 如果 beta 调度是 squaredcos_cap_v2，使用 Glide 的余弦调度
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果调度类型不被实现，抛出错误
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 如果需要，重新缩放 beta 值以适应零 SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        # 计算 alphas，1 - betas
        self.alphas = 1.0 - self.betas
        # 计算 alphas 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 在每个 ddim 步骤中，查看前一个 alphas_cumprod
        # 对于最终步骤，因为已经在 0，所以没有前一个 alphas_cumprod
        # `set_alpha_to_one` 决定是将该参数设置为 1 还是使用“非前一个”的最终 alpha
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的值
        self.num_inference_steps = None
        # 创建时间步张量，从 num_train_timesteps 逆序生成
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    # 定义输入样本的缩放函数，支持时间步长的可互换性
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器之间的可互换性。

        参数：
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。

        返回：
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 返回原始样本，当前未进行任何缩放
        return sample

    # 获取给定时间步和前一步的方差
    def _get_variance(self, timestep, prev_timestep):
        # 计算当前时间步的 alphas 累积乘积
        alpha_prod_t = self.alphas_cumprod[timestep]
        # 计算前一步的 alphas 累积乘积
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        # 计算当前和前一步的 beta 值
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 计算方差
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # 返回计算得到的方差
        return variance

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制而来
    # 定义动态阈值采样方法，输入为一个张量，输出为处理后的张量
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "动态阈值：在每个采样步骤中，我们将 s 设置为 xt0 中某个百分位绝对像素值（x_0 在时间步 t 的预测），
        如果 s > 1，则将 xt0 阈值限制在 [-s, s] 范围内，然后除以 s。动态阈值通过将饱和像素（接近 -1 和 1 的像素）向内推
        进，从而在每个步骤中积极防止像素饱和。我们发现动态阈值显著提高了照片真实感以及图像与文本的对齐，特别是在使用
        非常大的引导权重时。"
        https://arxiv.org/abs/2205.11487
        """
        # 获取输入样本的数据类型
        dtype = sample.dtype
        # 获取输入样本的批次大小、通道数和剩余维度
        batch_size, channels, *remaining_dims = sample.shape
    
        # 如果数据类型不是 float32 或 float64，则转换为 float，以便进行百分位计算
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 用于量化计算的上溯，且未实现对 cpu 半精度的夹紧
    
        # 将样本展平以便在每张图像上进行百分位计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
    
        # 计算样本的绝对值
        abs_sample = sample.abs()  # "某个百分位绝对像素值"
    
        # 计算样本绝对值的给定百分位
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 将 s 限制在指定范围内
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当最小值限制为 1 时，相当于标准限制在 [-1, 1]
        # 扩展 s 的维度，以便与样本广播
        s = s.unsqueeze(1)  # (batch_size, 1) 因为 clamp 将沿 dim=0 广播
        # 将样本限制在 [-s, s] 范围内并进行归一化
        sample = torch.clamp(sample, -s, s) / s  # "我们将 xt0 阈值限制在 [-s, s] 范围内并除以 s"
    
        # 将样本重新调整回原始形状
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原始数据类型
        sample = sample.to(dtype)
    
        # 返回处理后的样本
        return sample
    # 设置用于扩散链的离散时间步数（在推理之前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置生成样本时用于扩散步骤的离散时间步数。
    
        参数:
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数量。
        """
    
        # 检查推理步骤数是否超过训练时间步数
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} 不能大于 `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} 因为使用此调度器训练的unet模型只能处理"
                f" 最大 {self.config.num_train_timesteps} 个时间步。"
            )
    
        # 设置实例的推理步骤数
        self.num_inference_steps = num_inference_steps
    
        # 根据配置的时间步间隔方式生成时间步
        if self.config.timestep_spacing == "linspace":
            # 生成线性间隔的时间步，并反向排序
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            # 计算步长比率
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # 通过乘以步长比率生成整数时间步
            # 转换为整数以避免在推理步骤为3的幂时出现问题
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            # 添加偏移量
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            # 计算步长比率
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # 通过乘以步长比率生成整数时间步
            # 转换为整数以避免在推理步骤为3的幂时出现问题
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            # 减去1以调整时间步
            timesteps -= 1
        else:
            # 抛出不支持的时间步间隔类型的错误
            raise ValueError(
                f"{self.config.timestep_spacing} 不受支持。请确保选择 'leading' 或 'trailing'。"
            )
    
        # 将生成的时间步转换为张量并移动到指定设备
        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    # 定义推理步骤
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 与 original_samples 拥有相同的设备和数据类型
        # 将 self.alphas_cumprod 移动到指定设备，以避免后续 add_noise 调用时冗余的 CPU 到 GPU 数据传输
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为 original_samples 的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到 original_samples 的设备
        timesteps = timesteps.to(original_samples.device)

        # 计算 timesteps 对应的 sqrt_alpha_prod
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 original_samples，则在最后添加一个维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 timesteps 对应的 sqrt_one_minus_alpha_prod
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 original_samples，则在最后添加一个维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算带噪声的样本 noisy_samples
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回带噪声的样本
        return noisy_samples

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity 复制
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 与 sample 拥有相同的设备和数据类型
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        # 将 alphas_cumprod 转换为 sample 的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        # 将 timesteps 移动到 sample 的设备
        timesteps = timesteps.to(sample.device)

        # 计算 timesteps 对应的 sqrt_alpha_prod
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 sample，则在最后添加一个维度
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 timesteps 对应的 sqrt_one_minus_alpha_prod
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 sample，则在最后添加一个维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算速度 velocity
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回计算得到的速度
        return velocity

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```