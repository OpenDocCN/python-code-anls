# `.\diffusers\schedulers\scheduling_ddpm.py`

```py
# 版权声明，表明版权所有者和许可信息
# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权； 
# 您只能在遵守许可证的情况下使用此文件。
# 您可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件在许可证下分发是基于“按现状”原则， 
# 不提供任何形式的担保或条件，无论是明示还是暗示。
# 有关许可证的特定条款和条件，请参见许可证。

# 免责声明：此文件受到 https://github.com/ermongroup/ddim 的强烈影响

# 导入数学模块
import math
# 从 dataclasses 导入 dataclass 装饰器
from dataclasses import dataclass
# 导入类型注解
from typing import List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 PyTorch 库
import torch

# 导入配置相关的工具和注册功能
from ..configuration_utils import ConfigMixin, register_to_config
# 导入基本输出工具
from ..utils import BaseOutput
# 导入生成随机张量的工具
from ..utils.torch_utils import randn_tensor
# 导入调度相关的工具
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

# 定义调度器输出的数据类
@dataclass
class DDPMSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 用于图像)：
            先前时间步的计算样本 `(x_{t-1})`。 `prev_sample` 应作为下一个模型输入使用
            在去噪循环中。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 用于图像)：
            基于当前时间步模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进展或提供指导。
    """

    # 定义上一个样本的张量
    prev_sample: torch.Tensor
    # 可选的预测原始样本的张量
    pred_original_sample: Optional[torch.Tensor] = None

# 定义用于生成 beta 的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建一个 beta 计划，该计划离散化给定的 alpha_t_bar 函数，该函数定义了
    (1-beta) 在时间上的累积乘积，从 t = [0,1]。

    包含一个 alpha_bar 函数，该函数接受参数 t 并将其转换为 (1-beta) 的累积乘积
    直到扩散过程的那一部分。

    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta；使用小于 1 的值以
                     防止奇异性。
        alpha_transform_type (`str`, *可选*，默认为 `cosine`): alpha_bar 的噪声调度类型。
                     选择 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于对模型输出进行步骤的 betas
    """
    # 判断 alpha 变换类型是否为 "cosine"
    if alpha_transform_type == "cosine":
        # 定义 alpha_bar 函数，接受参数 t
        def alpha_bar_fn(t):
            # 返回基于余弦函数的转换值
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    # 检查 alpha_transform_type 是否为 "exp"
        elif alpha_transform_type == "exp":
            # 定义一个函数，根据输入 t 返回指数衰减值
            def alpha_bar_fn(t):
                return math.exp(t * -12.0)
    
        # 如果 alpha_transform_type 不支持，则抛出异常
        else:
            raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    
        # 初始化一个空列表 betas，用于存储 beta 值
        betas = []
        # 循环 num_diffusion_timesteps 次以计算 beta 值
        for i in range(num_diffusion_timesteps):
            # 计算当前时间步 t1
            t1 = i / num_diffusion_timesteps
            # 计算下一个时间步 t2
            t2 = (i + 1) / num_diffusion_timesteps
            # 计算 beta 值并添加到 betas 列表，限制最大值为 max_beta
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        # 将 betas 转换为张量并返回
        return torch.tensor(betas, dtype=torch.float32)
# 从 diffusers.schedulers.scheduling_ddim 导入 rescale_zero_terminal_snr 函数
def rescale_zero_terminal_snr(betas):
    """
    将 betas 重缩放为具有零终端 SNR，基于 https://arxiv.org/pdf/2305.08891.pdf (算法 1)

    参数:
        betas (`torch.Tensor`):
            初始化调度器时使用的 betas。

    返回:
        `torch.Tensor`: 重缩放后的 betas，具有零终端 SNR
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 开平方以得到 alphas_bar_sqrt

    # 存储旧值。
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 复制第一个值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 复制最后一个值

    # 移动，使最后时间步为零。
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 从每个值中减去最后一个值

    # 缩放，使第一个时间步恢复到旧值。
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 计算缩放因子

    # 将 alphas_bar_sqrt 转换回 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 还原累积乘积
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个值与其余部分连接
    betas = 1 - alphas  # 计算 betas

    return betas  # 返回计算后的 betas


class DDPMScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDPMScheduler` 探索去噪得分匹配与 Langevin 动力学采样之间的关系。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。查看超类文档以了解库为所有调度器实现的通用
    方法，如加载和保存。
    # 参数定义部分，描述各参数的作用和默认值
        Args:
            num_train_timesteps (`int`, defaults to 1000):  # 训练模型的扩散步骤数，默认值为1000
                The number of diffusion steps to train the model.  # 描述参数的作用
            
            beta_start (`float`, defaults to 0.0001):  # 推理的起始`beta`值，默认值为0.0001
                The starting `beta` value of inference.  # 描述参数的作用
                
            beta_end (`float`, defaults to 0.02):  # 最终的`beta`值，默认值为0.02
                The final `beta` value.  # 描述参数的作用
                
            beta_schedule (`str`, defaults to `"linear"`):  # `beta`调度方式，默认为“线性”
                The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from  # 描述参数的作用
                
                `linear`, `scaled_linear`, or `squaredcos_cap_v2`.  # 可选的调度方式
            
            trained_betas (`np.ndarray`, *optional*):  # 直接传递给构造函数的`beta`数组，可选参数
                An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.  # 描述参数的作用
                
            variance_type (`str`, defaults to `"fixed_small"`):  # 添加噪声时裁剪方差的类型，默认为“固定小”
                Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,  # 描述参数的作用
                
                `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.  # 可选的方差类型
            
            clip_sample (`bool`, defaults to `True`):  # 用于数值稳定性的样本裁剪，默认为True
                Clip the predicted sample for numerical stability.  # 描述参数的作用
                
            clip_sample_range (`float`, defaults to 1.0):  # 样本裁剪的最大幅度，默认为1.0，仅在`clip_sample=True`时有效
                The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.  # 描述参数的作用
                
            prediction_type (`str`, defaults to `epsilon`, *optional*):  # 调度函数的预测类型，默认为“epsilon”
                Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),  # 描述参数的作用
                
                `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen  # 可选的预测类型
                
                Video](https://imagen.research.google/video/paper.pdf) paper).  # 引用文献
            
            thresholding (`bool`, defaults to `False`):  # 是否使用“动态阈值”方法，默认为False
                Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such  # 描述参数的作用
                
                as Stable Diffusion.  # 说明不适用的模型
            
            dynamic_thresholding_ratio (`float`, defaults to 0.995):  # 动态阈值方法的比率，默认为0.995，仅在`thresholding=True`时有效
                The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.  # 描述参数的作用
                
            sample_max_value (`float`, defaults to 1.0):  # 动态阈值的阈值，默认为1.0，仅在`thresholding=True`时有效
                The threshold value for dynamic thresholding. Valid only when `thresholding=True`.  # 描述参数的作用
                
            timestep_spacing (`str`, defaults to `"leading"`):  # 时间步的缩放方式，默认为“领先”
                The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and  # 描述参数的作用
                
                Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.  # 引用文献
            
            steps_offset (`int`, defaults to 0):  # 推理步骤的偏移量，默认为0
                An offset added to the inference steps, as required by some model families.  # 描述参数的作用
                
            rescale_betas_zero_snr (`bool`, defaults to `False`):  # 是否重新缩放`beta`以使终端SNR为零，默认为False
                Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and  # 描述参数的作用
                
                dark samples instead of limiting it to samples with medium brightness. Loosely related to  # 说明模型的功能
                
                [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).  # 引用文献
    # 从 KarrasDiffusionSchedulers 中提取每个调度器的名称，生成一个兼容的列表
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置顺序为 1
    order = 1

    # 装饰器，将该方法注册到配置中
    @register_to_config
    def __init__(
        # 定义训练时间步数，默认为 1000
        num_train_timesteps: int = 1000,
        # 定义 beta 开始值，默认为 0.0001
        beta_start: float = 0.0001,
        # 定义 beta 结束值，默认为 0.02
        beta_end: float = 0.02,
        # 定义 beta 调度方式，默认为 "linear"
        beta_schedule: str = "linear",
        # 可选参数，训练好的 beta 值，可以是数组或列表
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 定义方差类型，默认为 "fixed_small"
        variance_type: str = "fixed_small",
        # 定义是否裁剪样本，默认为 True
        clip_sample: bool = True,
        # 定义预测类型，默认为 "epsilon"
        prediction_type: str = "epsilon",
        # 定义是否使用阈值处理，默认为 False
        thresholding: bool = False,
        # 定义动态阈值处理的比例，默认为 0.995
        dynamic_thresholding_ratio: float = 0.995,
        # 定义样本裁剪范围，默认为 1.0
        clip_sample_range: float = 1.0,
        # 定义样本最大值，默认为 1.0
        sample_max_value: float = 1.0,
        # 定义时间步间距方式，默认为 "leading"
        timestep_spacing: str = "leading",
        # 定义时间步偏移量，默认为 0
        steps_offset: int = 0,
        # 定义是否为零 SNR 重缩放，默认为 False
        rescale_betas_zero_snr: bool = False,
    ):
        # 如果提供了训练好的 beta 值，则将其转换为张量
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果 beta 调度为线性，则生成线性 beta 序列
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果 beta 调度为 scaled_linear，则生成特定的 beta 序列
        elif beta_schedule == "scaled_linear":
            # 该调度特定于潜在扩散模型
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 如果 beta 调度为 squaredcos_cap_v2，则使用 Glide 余弦调度生成 beta 序列
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果 beta 调度为 sigmoid，则生成 Sigmoid 调度的 beta 序列
        elif beta_schedule == "sigmoid":
            # GeoDiff Sigmoid 调度
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        # 如果 beta 调度方式不被实现，抛出未实现错误
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 如果选择了零 SNR 重缩放，则进行重缩放处理
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        # 计算 alpha 值，alpha = 1 - beta
        self.alphas = 1.0 - self.betas
        # 计算累积的 alpha 值
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 创建一个包含 1.0 的张量
        self.one = torch.tensor(1.0)

        # 设置初始噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的值，表示是否自定义时间步
        self.custom_timesteps = False
        # 推理步骤的数量，初始为 None
        self.num_inference_steps = None
        # 创建一个反向的时间步序列
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        # 设置方差类型
        self.variance_type = variance_type

    # 定义缩放模型输入的方法
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器之间的互换性。

        参数：
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。

        返回：
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 返回未缩放的输入样本
        return sample
    # 设置推断步骤和设备等参数的函数
        def set_timesteps(
            # 可选的推断步骤数
            self,
            num_inference_steps: Optional[int] = None,
            # 可选的设备类型，字符串或 torch.device
            device: Union[str, torch.device] = None,
            # 可选的时间步列表
            timesteps: Optional[List[int]] = None,
        # 获取方差的函数
        def _get_variance(self, t, predicted_variance=None, variance_type=None):
            # 获取当前时间步的前一个时间步
            prev_t = self.previous_timestep(t)
    
            # 获取当前时间步的累积 alpha 值
            alpha_prod_t = self.alphas_cumprod[t]
            # 获取前一个时间步的累积 alpha 值，如果没有则使用 one
            alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
            # 计算当前时间步的 beta 值
            current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
    
            # 计算预测的方差，依据文献中的公式
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
    
            # 确保方差的对数不为零，限制其最小值
            variance = torch.clamp(variance, min=1e-20)
    
            # 如果未指定方差类型，则使用配置中的方差类型
            if variance_type is None:
                variance_type = self.config.variance_type
    
            # 针对训练稳定性的一些调整
            if variance_type == "fixed_small":
                variance = variance
            # 针对 rl-diffuser 的特定处理
            elif variance_type == "fixed_small_log":
                variance = torch.log(variance)
                variance = torch.exp(0.5 * variance)
            elif variance_type == "fixed_large":
                variance = current_beta_t
            elif variance_type == "fixed_large_log":
                # Glide 的最大对数方差
                variance = torch.log(current_beta_t)
            elif variance_type == "learned":
                # 返回预测的方差
                return predicted_variance
            elif variance_type == "learned_range":
                # 计算最小和最大对数方差，并根据预测方差加权
                min_log = torch.log(variance)
                max_log = torch.log(current_beta_t)
                frac = (predicted_variance + 1) / 2
                variance = frac * max_log + (1 - frac) * min_log
    
            # 返回最终的方差值
            return variance
    # 定义私有方法，进行动态阈值采样，输入为样本张量，输出为处理后的张量
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "动态阈值处理：在每个采样步骤中，我们将 s 设置为 xt0 中某个百分位的绝对像素值（t 时刻对 x_0 的预测），
        如果 s > 1，则将 xt0 限制在 [-s, s] 范围内，然后除以 s。动态阈值处理将饱和像素（接近 -1 和 1 的像素）
        向内推，使每一步都能主动防止像素饱和。我们发现动态阈值处理显著提高了真实感以及图像与文本的对齐，
        尤其是在使用非常大的引导权重时。"
        
        https://arxiv.org/abs/2205.11487
        """
        # 获取样本的数据类型
        dtype = sample.dtype
        # 获取样本的批量大小、通道数以及剩余维度
        batch_size, channels, *remaining_dims = sample.shape

        # 如果数据类型不是 float32 或 float64，则转换为 float
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 为了进行分位数计算而进行类型提升，且 clamp 对 CPU half 类型未实现

        # 将样本展平，以便在每个图像上进行分位数计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        # 计算样本的绝对值
        abs_sample = sample.abs()  # "某个百分位的绝对像素值"

        # 计算每个样本的动态阈值 s
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 将 s 限制在 min=1 和 max=self.config.sample_max_value 之间
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当被限制到 min=1 时，相当于标准的裁剪到 [-1, 1]
        # 在第一个维度上扩展 s 的维度，使其形状为 (batch_size, 1)
        s = s.unsqueeze(1)  # (batch_size, 1)，因为 clamp 会在维度 0 上广播
        # 将样本裁剪到 [-s, s] 范围内，然后除以 s
        sample = torch.clamp(sample, -s, s) / s  # "我们将 xt0 限制在 [-s, s] 范围内，然后除以 s"

        # 将样本恢复到原来的形状
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原来的数据类型
        sample = sample.to(dtype)

        # 返回处理后的样本
        return sample

    # 定义步进方法，输入为模型输出、时间步、样本、生成器和返回字典的标志
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    # 定义添加噪声的方法，输入为原始样本、噪声和时间步
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    # 返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 具有与 original_samples 相同的设备和数据类型
        # 将 self.alphas_cumprod 移动到设备上，以避免后续 add_noise 调用时重复的 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为与 original_samples 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到与 original_samples 相同的设备
        timesteps = timesteps.to(original_samples.device)
    
        # 计算平方根的 alpha 乘积
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将结果展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 在 sqrt_alpha_prod 的形状小于 original_samples 时，添加新维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
        # 计算平方根的 (1 - alpha) 乘积
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将结果展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 在 sqrt_one_minus_alpha_prod 的形状小于 original_samples 时，添加新维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
        # 生成带噪声的样本
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回带噪声的样本
        return noisy_samples
    
    # 定义获取速度的函数，输入为样本、噪声和时间步
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 具有与 sample 相同的设备和数据类型
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        # 将 alphas_cumprod 转换为与 sample 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        # 将 timesteps 移动到与 sample 相同的设备
        timesteps = timesteps.to(sample.device)
    
        # 计算平方根的 alpha 乘积
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将结果展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 在 sqrt_alpha_prod 的形状小于 sample 时，添加新维度
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
        # 计算平方根的 (1 - alpha) 乘积
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将结果展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 在 sqrt_one_minus_alpha_prod 的形状小于 sample 时，添加新维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
        # 计算速度
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回计算得到的速度
        return velocity
    
    # 定义获取对象长度的函数
    def __len__(self):
        # 返回配置中的训练时间步数
        return self.config.num_train_timesteps
    
    # 定义获取上一个时间步的函数
    def previous_timestep(self, timestep):
        # 如果使用自定义时间步
        if self.custom_timesteps:
            # 找到当前时间步的索引
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            # 如果当前索引是最后一个，返回 -1
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                # 返回下一个时间步
                prev_t = self.timesteps[index + 1]
        else:
            # 计算推理步骤数
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
            )
            # 返回上一个时间步
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps
    
        # 返回上一个时间步
        return prev_t
```