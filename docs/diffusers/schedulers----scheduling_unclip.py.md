# `.\diffusers\schedulers\scheduling_unclip.py`

```py
# 版权所有 2024 Kakao Brain 和 HuggingFace Team。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件在许可证下分发均为“按原样”提供，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 请参见许可证以获取与权限和
# 限制相关的特定语言。

# 导入数学模块
import math
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入类型提示相关的模块
from typing import Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch

# 从配置工具导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入 BaseOutput
from ..utils import BaseOutput
# 从 torch_utils 导入 randn_tensor 函数
from ..utils.torch_utils import randn_tensor
# 从调度工具导入 SchedulerMixin
from .scheduling_utils import SchedulerMixin


@dataclass
# 从 diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput 复制的类，包含 DDPM 到 UnCLIP 的转换
class UnCLIPSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            计算出的前一个时间步的样本 `(x_{t-1})`。 `prev_sample` 应在去噪循环中作为下一个模型输入使用。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            基于当前时间步的模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或进行引导。
    """

    # 上一个样本的张量
    prev_sample: torch.Tensor
    # 可选的预测去噪样本
    pred_original_sample: Optional[torch.Tensor] = None


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建一个 beta 计划，该计划离散化给定的 alpha_t_bar 函数，该函数定义了 (1-beta) 随时间的累积乘积
    从 t = [0,1] 开始。

    包含一个 alpha_bar 函数，该函数接受参数 t，并将其转换为扩散过程至该部分的 (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以
                     防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认值为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择。

    返回：
        betas (`np.ndarray`): 调度器用于更新模型输出的 betas
    """
    # 检查 alpha 转换类型是否为 "cosine"
    if alpha_transform_type == "cosine":
        # 定义 alpha_bar 函数，接受参数 t
        def alpha_bar_fn(t):
            # 计算 t 的 alpha_bar 值，使用余弦函数
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    # 检查 alpha_transform_type 是否为 "exp"
    elif alpha_transform_type == "exp":
        
        # 定义 alpha_bar_fn 函数，用于计算指数衰减
        def alpha_bar_fn(t):
            # 计算并返回指数衰减值
            return math.exp(t * -12.0)

    # 如果 alpha_transform_type 不是已知类型，抛出异常
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化空列表 betas 用于存储每个时间步的 beta 值
    betas = []
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步的归一化值 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步的归一化值 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值并添加到 betas 列表，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 betas 转换为张量并返回
    return torch.tensor(betas, dtype=torch.float32)
# 定义 UnCLIPScheduler 类，继承自 SchedulerMixin 和 ConfigMixin
class UnCLIPScheduler(SchedulerMixin, ConfigMixin):
    """
    NOTE: do not use this scheduler. The DDPM scheduler has been updated to support the changes made here. This
    scheduler will be removed and replaced with DDPM.

    This is a modified DDPM Scheduler specifically for the karlo unCLIP model.

    This scheduler has some minor variations in how it calculates the learned range variance and dynamically
    re-calculates betas based off the timesteps it is skipping.

    The scheduler also uses a slightly different step ratio when computing timesteps to use for inference.

    See [`~DDPMScheduler`] for more information on DDPM scheduling

    Args:
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        variance_type (`str`):
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small_log`
            or `learned_range`.
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between `-clip_sample_range` and `clip_sample_range` for numerical
            stability.
        clip_sample_range (`float`, default `1.0`):
            The range to clip the sample between. See `clip_sample`.
        prediction_type (`str`, default `epsilon`, optional):
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process)
            or `sample` (directly predicting the noisy sample`)
    """

    # 注册构造函数到配置中
    @register_to_config
    def __init__(
        # 定义训练时间步的数量，默认为 1000
        self,
        num_train_timesteps: int = 1000,
        # 定义方差类型，默认为 'fixed_small_log'
        variance_type: str = "fixed_small_log",
        # 是否裁剪样本，默认为 True
        clip_sample: bool = True,
        # 样本裁剪范围，默认为 1.0
        clip_sample_range: Optional[float] = 1.0,
        # 预测类型，默认为 'epsilon'
        prediction_type: str = "epsilon",
        # β 调度方式，默认为 'squaredcos_cap_v2'
        beta_schedule: str = "squaredcos_cap_v2",
    ):
        # 检查 β 调度方式是否有效
        if beta_schedule != "squaredcos_cap_v2":
            raise ValueError("UnCLIPScheduler only supports `beta_schedule`: 'squaredcos_cap_v2'")

        # 根据训练时间步计算 beta 值
        self.betas = betas_for_alpha_bar(num_train_timesteps)

        # 计算 alpha 值
        self.alphas = 1.0 - self.betas
        # 计算累积的 alpha 值
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 初始化一个值为 1.0 的张量
        self.one = torch.tensor(1.0)

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的推理步数
        self.num_inference_steps = None
        # 创建时间步张量，从 num_train_timesteps 到 0 递减
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        # 设置方差类型
        self.variance_type = variance_type

    # 定义缩放模型输入的方法
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.Tensor`: scaled input sample
        """
        # 返回未处理的样本，未进行缩放
        return sample
    # 定义设置离散时间步长的方法，用于扩散链
        def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
            """
            设置用于扩散链的离散时间步长。此方法应在推理前运行。
    
            注意：此调度器使用与其他扩散调度器略有不同的步长比例。
            不同的步长比例是为了模仿原始的 karlo 实现，并不影响结果的质量或准确性。
    
            参数:
                num_inference_steps (`int`):
                    生成样本时使用的扩散步数。
            """
            # 将输入的推理步骤数量保存到实例变量
            self.num_inference_steps = num_inference_steps
            # 计算步长比例
            step_ratio = (self.config.num_train_timesteps - 1) / (self.num_inference_steps - 1)
            # 生成时间步长数组，反向排列并转换为整数类型
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            # 将时间步长转换为张量并移动到指定设备
            self.timesteps = torch.from_numpy(timesteps).to(device)
    
        # 定义获取方差的方法，计算当前和前一个时间步的方差
        def _get_variance(self, t, prev_timestep=None, predicted_variance=None, variance_type=None):
            # 如果没有提供前一个时间步，默认为当前时间步减一
            if prev_timestep is None:
                prev_timestep = t - 1
    
            # 获取当前时间步和前一个时间步的累积 alpha 值
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
            # 计算当前和前一个时间步的 beta 值
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
    
            # 根据前后时间步计算 beta 值
            if prev_timestep == t - 1:
                beta = self.betas[t]
            else:
                beta = 1 - alpha_prod_t / alpha_prod_t_prev
    
            # 对于 t > 0，计算预测方差
            # x_{t-1} ~ N(pred_prev_sample, variance) == 将方差添加到预测样本中
            variance = beta_prod_t_prev / beta_prod_t * beta
    
            # 如果没有提供方差类型，则使用配置中的类型
            if variance_type is None:
                variance_type = self.config.variance_type
    
            # 针对训练稳定性进行的一些特殊处理
            if variance_type == "fixed_small_log":
                # 将方差限制到最小值，然后计算对数
                variance = torch.log(torch.clamp(variance, min=1e-20))
                # 计算最终方差
                variance = torch.exp(0.5 * variance)
            elif variance_type == "learned_range":
                # 注意与 DDPM 调度器的区别
                min_log = variance.log()
                max_log = beta.log()
    
                # 计算方差的比例
                frac = (predicted_variance + 1) / 2
                # 计算最终方差
                variance = frac * max_log + (1 - frac) * min_log
    
            # 返回计算得到的方差
            return variance
    
        # 定义步骤方法，处理模型输出和样本
        def step(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            prev_timestep: Optional[int] = None,
            generator=None,
            return_dict: bool = True,
        # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的方法
        def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.IntTensor,
    # 返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 与 original_samples 具有相同的设备和数据类型
        # 将 self.alphas_cumprod 移动到相应的设备，以避免后续 add_noise 调用中的冗余 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为 original_samples 的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到 original_samples 的设备
        timesteps = timesteps.to(original_samples.device)
    
        # 计算 alphas_cumprod 在 timesteps 位置的平方根
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将结果展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 original_samples 的维度，则添加新的维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
        # 计算 (1 - alphas_cumprod) 在 timesteps 位置的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将结果展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 original_samples 的维度，则添加新的维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
        # 计算带噪声的样本
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回带噪声的样本
        return noisy_samples
```