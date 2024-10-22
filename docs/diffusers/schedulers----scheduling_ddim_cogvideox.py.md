# `.\diffusers\schedulers\scheduling_ddim_cogvideox.py`

```py
# 版权所有 2024 The CogVideoX 团队，清华大学和 ZhipuAI 及 HuggingFace 团队。
# 保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则不得使用此文件。
# 您可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议，否则根据许可证分发的软件
# 是以“按现状”基础分发的，不提供任何形式的保证或条件，
# 无论是明示或暗示的。有关许可证下特定语言的权限和
# 限制，请参阅许可证。

# 免责声明：此代码深受 https://github.com/pesser/pytorch_diffusion
# 和 https://github.com/hojonathanho/diffusion 的影响

# 导入数学库
import math
# 从数据类模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从类型模块导入 List、Optional、Tuple 和 Union 类型
from typing import List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 torch 库
import torch

# 从配置工具模块导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入 BaseOutput 类
from ..utils import BaseOutput
# 从调度工具模块导入 KarrasDiffusionSchedulers 和 SchedulerMixin
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
# 从 diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput 复制，进行 DDPM 到 DDIM 的转换
class DDIMSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像)：
            计算得出的前一时间步的样本 `(x_{t-1})`。`prev_sample` 应作为下一个模型输入
            在去噪循环中使用。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像)：
            基于当前时间步模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或指导。
    """

    # 定义前一个样本的张量
    prev_sample: torch.Tensor
    # 可选的预测原始样本的张量
    pred_original_sample: Optional[torch.Tensor] = None


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建一个 beta 调度，该调度离散化给定的 alpha_t_bar 函数，该函数定义了
    随着时间推移 (1-beta) 的累积乘积，从 t = [0,1]。

    包含一个函数 alpha_bar，该函数接受参数 t 并将其转换为
    在扩散过程中到目前为止 (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 要生成的 beta 数量。
        max_beta (`float`): 要使用的最大 beta 值；使用低于 1 的值以
                     防止奇异性。
        alpha_transform_type (`str`, *可选*，默认为 `cosine`): alpha_bar 的噪声调度类型。
                     选择 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于更新模型输出的 betas
    """
    # 检查 alpha_transform_type 是否为 "cosine"
        if alpha_transform_type == "cosine":
            # 定义 alpha_bar_fn 函数，用于计算基于余弦的 alpha 值
            def alpha_bar_fn(t):
                # 返回余弦函数的平方，调整参数以适应时间 t
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    
        # 检查 alpha_transform_type 是否为 "exp"
        elif alpha_transform_type == "exp":
            # 定义 alpha_bar_fn 函数，用于计算基于指数的 alpha 值
            def alpha_bar_fn(t):
                # 返回以 t 为输入的指数衰减值
                return math.exp(t * -12.0)
    
        # 如果 alpha_transform_type 既不是 "cosine" 也不是 "exp"
        else:
            # 抛出不支持的 alpha_transform_type 异常
            raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    
        # 初始化一个空列表 betas，用于存储 beta 值
        betas = []
        # 遍历每一个扩散时间步
        for i in range(num_diffusion_timesteps):
            # 计算当前时间步 t1
            t1 = i / num_diffusion_timesteps
            # 计算下一个时间步 t2
            t2 = (i + 1) / num_diffusion_timesteps
            # 将计算得到的 beta 值添加到列表中
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        # 返回 betas 列表作为一个张量，数据类型为 float32
        return torch.tensor(betas, dtype=torch.float32)
# 定义函数以根据给定的累积 alpha 值重新调整 beta 值，使最终信噪比为零
def rescale_zero_terminal_snr(alphas_cumprod):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """

    # 计算累积 alpha 值的平方根
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # 存储旧的值，以便后续计算
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # 将最后一个时间步的值移至零
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # 调整比例，使第一个时间步的值恢复为旧值
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # 将 alphas_bar_sqrt 转换为 beta 值
    alphas_bar = alphas_bar_sqrt**2  # 恢复平方

    # 返回调整后的 beta 值
    return alphas_bar


# 定义类以扩展去噪过程，采用非马尔可夫指导
class CogVideoXDDIMScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDIMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    """

    # 兼容的调度器名称列表
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 调度器的顺序
    order = 1

    # 装饰器注册配置
    @register_to_config
    def __init__(
        # 训练时间步的数量，默认为1000
        num_train_timesteps: int = 1000,
        # beta 的起始值
        beta_start: float = 0.00085,
        # beta 的结束值
        beta_end: float = 0.0120,
        # beta 的调度类型
        beta_schedule: str = "scaled_linear",
        # 经过训练的 beta 值，可选参数
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 是否裁剪样本
        clip_sample: bool = True,
        # 是否将 alpha 设置为 1
        set_alpha_to_one: bool = True,
        # 时间步的偏移量
        steps_offset: int = 0,
        # 预测类型
        prediction_type: str = "epsilon",
        # 裁剪样本的范围
        clip_sample_range: float = 1.0,
        # 样本的最大值
        sample_max_value: float = 1.0,
        # 时间步的间隔类型
        timestep_spacing: str = "leading",
        # 是否重新调整 beta 以实现零信噪比
        rescale_betas_zero_snr: bool = False,
        # 信噪比的偏移缩放因子
        snr_shift_scale: float = 3.0,
    ):
        # 检查是否提供了经过训练的 beta 值
        if trained_betas is not None:
            # 将训练的 beta 转换为张量，数据类型为 float32
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果 beta 调度为线性
        elif beta_schedule == "linear":
            # 生成一个线性空间的 beta 值
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果 beta 调度为 scaled_linear
        elif beta_schedule == "scaled_linear":
            # 此调度非常特定于潜在扩散模型
            # 生成平方根后的线性空间，并再平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
        # 如果 beta 调度为 squaredcos_cap_v2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            # 抛出未实现错误
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alpha 值为 1 减去 beta
        self.alphas = 1.0 - self.betas
        # 计算 alpha 累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 根据 SNR shift 进行修改，遵循 SD3
        self.alphas_cumprod = self.alphas_cumprod / (snr_shift_scale + (1 - snr_shift_scale) * self.alphas_cumprod)

        # 如果需要对零 SNR 进行重缩放
        if rescale_betas_zero_snr:
            # 重缩放到零终端 SNR
            self.alphas_cumprod = rescale_zero_terminal_snr(self.alphas_cumprod)

        # 在每一步中，我们查看之前的 alphas_cumprod
        # 最后一步时，没有前一个 alphas_cumprod，因为已经到达 0
        # `set_alpha_to_one` 决定是否将此参数设置为 1 或使用最终 alpha
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的值
        self.num_inference_steps = None
        # 创建反向时间步长的张量
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    def _get_variance(self, timestep, prev_timestep):
        # 获取当前时间步的 alpha 累积乘积
        alpha_prod_t = self.alphas_cumprod[timestep]
        # 获取前一个时间步的 alpha 累积乘积
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        # 计算当前和前一个时间步的 beta 值
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 根据 beta 值计算方差
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # 返回计算出的方差
        return variance

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器互换性。

        Args:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。

        Returns:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 返回未缩放的样本（示例中未实际实现缩放）
        return sample
    # 设置离散时间步，用于扩散链（在推理前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于生成样本的扩散步骤数量。
    
        参数:
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数量。
        """
    
        # 检查推理步骤是否超过训练时步数的最大值
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} 不能大于 `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps}，因为训练此调度器的unet模型只能处理"
                f" 最大 {self.config.num_train_timesteps} 步数。"
            )
    
        # 将推理步骤数量赋值给实例变量
        self.num_inference_steps = num_inference_steps
    
        # 根据配置中的时间步间隔类型计算时间步
        if self.config.timestep_spacing == "linspace":
            # 生成从0到训练时间步数-1的均匀间隔时间步
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            # 计算每步的比例，用于生成整数量的时间步
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # 创建整数时间步，通过比例乘法得到
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            # 计算每步的比例，用于生成整数量的时间步
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # 生成整数量时间步，通过比例乘法得到
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            # 如果时间步间隔不受支持，则抛出错误
            raise ValueError(
                f"{self.config.timestep_spacing} 不受支持。请确保选择 'leading' 或 'trailing' 之一。"
            )
    
        # 将计算得到的时间步转换为张量并移至指定设备
        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    # 步骤函数，处理模型输出和样本
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
    ):
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 中复制的函数
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 与 original_samples 具有相同的设备和数据类型
        # 将 self.alphas_cumprod 移动到设备，以避免后续 add_noise 调用中冗余的 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为与 original_samples 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到 original_samples 所在的设备
        timesteps = timesteps.to(original_samples.device)

        # 计算 alpha 的平方根乘积
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 original_samples，则在最后增加维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 (1 - alpha) 的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 original_samples，则在最后增加维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 生成噪声样本，结合原始样本和噪声
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回噪声样本
        return noisy_samples

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity 复制而来
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 与 sample 具有相同的设备和数据类型
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        # 将 alphas_cumprod 转换为与 sample 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        # 将 timesteps 移动到 sample 所在的设备
        timesteps = timesteps.to(sample.device)

        # 计算 alpha 的平方根乘积
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 sample，则在最后增加维度
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 (1 - alpha) 的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 sample，则在最后增加维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算速度，结合噪声和样本
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回计算出的速度
        return velocity

    # 获取训练时间步长的数量
    def __len__(self):
        # 返回训练时间步长的数量
        return self.config.num_train_timesteps
```