# `.\diffusers\schedulers\scheduling_lms_discrete.py`

```py
# 版权声明，列出版权所有者及其保留的权利
# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 你不得在未遵守许可证的情况下使用此文件。
# 你可以在以下网址获得许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件根据许可证分发，按"原样"提供，
# 不提供任何形式的明示或暗示的担保或条件。
# 详见许可证中关于权限和限制的具体条款。
import math  # 导入数学模块以使用数学函数
import warnings  # 导入警告模块以发出警告信息
from dataclasses import dataclass  # 导入数据类装饰器以简化类的定义
from typing import List, Optional, Tuple, Union  # 导入类型注解以增强代码可读性

import numpy as np  # 导入NumPy库以处理数组和数值计算
import torch  # 导入PyTorch库以进行张量计算
from scipy import integrate  # 从SciPy库导入积分功能

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入混合类和注册功能
from ..utils import BaseOutput  # 从工具模块导入基础输出类
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin  # 从调度工具导入相关类

@dataclass
# 从diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput复制，重命名为LMSDiscrete
class LMSDiscreteSchedulerOutput(BaseOutput):
    """
    调度器`step`函数输出的输出类。

    参数:
        prev_sample (`torch.Tensor`形状为`(batch_size, num_channels, height, width)`的图像):
            上一时间步的计算样本`(x_{t-1})`。`prev_sample`应作为下一个模型输入用于去噪循环。
        pred_original_sample (`torch.Tensor`形状为`(batch_size, num_channels, height, width)`的图像):
            基于当前时间步模型输出的预测去噪样本`(x_{0})`。
            `pred_original_sample`可用于预览进度或进行指导。
    """

    prev_sample: torch.Tensor  # 存储上一时间步的样本
    pred_original_sample: Optional[torch.Tensor] = None  # 存储预测的原始样本，默认为None


# 从diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar复制
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 传入的扩散时间步数
    max_beta=0.999,  # 最大的beta值，防止奇异性使用小于1的值
    alpha_transform_type="cosine",  # 噪声调度的类型，默认为"cosine"
):
    """
    创建一个beta调度器，该调度器离散化给定的alpha_t_bar函数，
    该函数定义了从t = [0,1]开始的(1-beta)的累积乘积。

    包含一个alpha_bar函数，该函数接受一个参数t，并将其转换为在扩散过程的这一部分
    中(1-beta)的累积乘积。

    参数:
        num_diffusion_timesteps (`int`): 要生成的beta数量。
        max_beta (`float`): 使用的最大beta值；使用小于1的值以防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为`cosine`): alpha_bar的噪声调度类型。
                     可选择`cosine`或`exp`

    返回:
        betas (`np.ndarray`): 调度器用于更新模型输出的beta值
    """
    if alpha_transform_type == "cosine":  # 检查alpha调度类型是否为"cosine"

        def alpha_bar_fn(t):  # 定义alpha_bar函数
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # 返回经过变换的alpha值
    # 检查 alpha_transform_type 是否为 "exp"
    elif alpha_transform_type == "exp":

        # 定义 alpha_bar_fn 函数，接受参数 t
        def alpha_bar_fn(t):
            # 计算指数衰减函数，返回 e 的 (t * -12.0) 次方
            return math.exp(t * -12.0)

    # 如果 alpha_transform_type 不是预期的类型，则抛出异常
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化一个空列表 betas，用于存储计算出的 beta 值
    betas = []
    # 遍历从 0 到 num_diffusion_timesteps - 1 的每个整数
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值并添加到列表，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 betas 列表转换为 PyTorch 的浮点张量并返回
    return torch.tensor(betas, dtype=torch.float32)
# 定义一个线性多步调度器类，继承自 SchedulerMixin 和 ConfigMixin
class LMSDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    一个用于离散 beta 计划的线性多步调度器。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查看超类文档以了解库为所有调度器实现的通用方法，如加载和保存。

    参数：
        num_train_timesteps (`int`, 默认值为 1000):
            用于训练模型的扩散步骤数量。
        beta_start (`float`, 默认值为 0.0001):
            推断的起始 `beta` 值。
        beta_end (`float`, 默认值为 0.02):
            最终的 `beta` 值。
        beta_schedule (`str`, 默认值为 `"linear"`):
            beta 计划，将 beta 范围映射到一系列用于模型步进的 betas。可以选择 `linear` 或 `scaled_linear`。
        trained_betas (`np.ndarray`, *可选*):
            直接将 beta 数组传递给构造函数，以绕过 `beta_start` 和 `beta_end`。
        use_karras_sigmas (`bool`, *可选*, 默认值为 `False`):
            是否在采样过程中使用 Karras sigmas 作为噪声计划中的步长。如果为 `True`，则根据噪声水平序列 {σi} 确定 sigmas。
        prediction_type (`str`, 默认值为 `epsilon`, *可选*):
            调度器函数的预测类型；可以是 `epsilon`（预测扩散过程的噪声）、`sample`（直接预测带噪声的样本）或 `v_prediction`（参见 [Imagen Video](https://imagen.research.google/video/paper.pdf) 论文的第 2.4 节）。
        timestep_spacing (`str`, 默认值为 `"linspace"`):
            时间步的缩放方式。有关更多信息，请参阅 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) 的表 2。
        steps_offset (`int`, 默认值为 0):
            添加到推断步骤的偏移量，某些模型系列需要该偏移量。
    """

    # 兼容的调度器名称列表，从 KarrasDiffusionSchedulers 获取
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置调度器的顺序
    order = 1

    # 用于注册配置的初始化方法
    @register_to_config
    def __init__(
        # 训练步骤数量，默认为 1000
        self,
        num_train_timesteps: int = 1000,
        # 起始 beta 值，默认为 0.0001
        beta_start: float = 0.0001,
        # 结束 beta 值，默认为 0.02
        beta_end: float = 0.02,
        # beta 计划，默认为 "linear"
        beta_schedule: str = "linear",
        # 经过训练的 beta 值，默认为 None
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 是否使用 Karras sigmas，默认为 False
        use_karras_sigmas: Optional[bool] = False,
        # 预测类型，默认为 "epsilon"
        prediction_type: str = "epsilon",
        # 时间步缩放方式，默认为 "linspace"
        timestep_spacing: str = "linspace",
        # 步骤偏移量，默认为 0
        steps_offset: int = 0,
    ):
        # 检查是否提供了训练好的贝塔值
        if trained_betas is not None:
            # 将训练好的贝塔值转换为 PyTorch 张量，数据类型为 float32
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 检查贝塔调度类型是否为线性
        elif beta_schedule == "linear":
            # 生成一个从 beta_start 到 beta_end 的线性序列，包含 num_train_timesteps 个值
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 检查贝塔调度类型是否为 scaled_linear
        elif beta_schedule == "scaled_linear":
            # 此调度特定于潜在扩散模型
            # 生成从 beta_start^0.5 到 beta_end^0.5 的线性序列，然后平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 检查贝塔调度类型是否为 squaredcos_cap_v2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            # 如果提供的调度类型未实现，抛出未实现错误
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alphas，等于 1 减去 betas
        self.alphas = 1.0 - self.betas
        # 计算 alphas 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 计算 sigmas，基于 alphas_cumprod 的公式
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        # 反转 sigmas 数组并添加一个 0.0 值
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        # 将 sigmas 转换为 PyTorch 张量
        self.sigmas = torch.from_numpy(sigmas)

        # 可设置的值
        self.num_inference_steps = None  # 推理步骤数初始化为 None
        self.use_karras_sigmas = use_karras_sigmas  # 使用 Karras sigmas 的标志
        # 设置时间步长
        self.set_timesteps(num_train_timesteps, None)
        self.derivatives = []  # 初始化导数列表
        self.is_scale_input_called = False  # 标志，表示是否调用过缩放输入

        self._step_index = None  # 当前步骤索引初始化为 None
        self._begin_index = None  # 起始步骤索引初始化为 None
        # 将 sigmas 移动到 CPU，避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  

    @property
    def init_noise_sigma(self):
        # 返回初始噪声分布的标准差
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            # 返回 sigmas 的最大值
            return self.sigmas.max()

        # 返回 sigmas 最大值的平方加 1 再开平方
        return (self.sigmas.max() ** 2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加 1。
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应该通过 `set_begin_index` 方法从管道设置。
        """
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制的代码
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的起始索引。此函数应该在推理之前从管道运行。

        参数：
            begin_index (`int`):
                调度器的起始索引。
        """
        self._begin_index = begin_index  # 设置起始索引
    # 定义一个方法，用于根据当前时间步缩放模型输入
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器的互换性。
    
        参数:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`float` or `torch.Tensor`):
                扩散链中的当前时间步。
    
        返回:
            `torch.Tensor`:
                缩放后的输入样本。
        """
    
        # 如果步索引为空，则初始化步索引
        if self.step_index is None:
            self._init_step_index(timestep)
    
        # 根据当前步索引获取相应的 sigma 值
        sigma = self.sigmas[self.step_index]
        # 将样本缩放，缩放因子为 sqrt(sigma^2 + 1)
        sample = sample / ((sigma**2 + 1) ** 0.5)
        # 标记输入缩放方法已被调用
        self.is_scale_input_called = True
        # 返回缩放后的样本
        return sample
    
    # 定义一个方法，用于计算线性多步系数
    def get_lms_coefficient(self, order, t, current_order):
        """
        计算线性多步系数。
    
        参数:
            order ():
            t ():
            current_order ():
        """
    
        # 定义一个内部函数，用于计算 LMS 导数
        def lms_derivative(tau):
            prod = 1.0
            # 遍历到 order 的每一个步骤
            for k in range(order):
                # 跳过当前阶数
                if current_order == k:
                    continue
                # 计算导数的乘积
                prod *= (tau - self.sigmas[t - k]) / (self.sigmas[t - current_order] - self.sigmas[t - k])
            # 返回导数的结果
            return prod
    
        # 通过数值积分计算集成系数，范围从 self.sigmas[t] 到 self.sigmas[t + 1]
        integrated_coeff = integrate.quad(lms_derivative, self.sigmas[t], self.sigmas[t + 1], epsrel=1e-4)[0]
    
        # 返回集成后的系数
        return integrated_coeff
    # 定义设置离散时间步长的方法，接受推理步骤数和设备参数
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步长（在推理之前运行）。
    
        参数：
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数。
            device (`str` 或 `torch.device`, *可选*):
                要将时间步长移动到的设备。如果为 `None`，则不移动时间步长。
        """
        # 将推理步骤数赋值给实例变量
        self.num_inference_steps = num_inference_steps
    
        # 根据配置的时间步长间隔选择不同的处理方式
        if self.config.timestep_spacing == "linspace":
            # 创建线性间隔的时间步长，并反转顺序
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[
                ::-1
            ].copy()
        elif self.config.timestep_spacing == "leading":
            # 计算步长比率
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # 创建整数时间步长，通过乘以比率生成
            # 转换为整数以避免 num_inference_step 为 3 的幂时出现问题
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            timesteps += self.config.steps_offset  # 添加偏移量
        elif self.config.timestep_spacing == "trailing":
            # 计算步长比率
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # 创建整数时间步长，通过乘以比率生成
            # 转换为整数以避免 num_inference_step 为 3 的幂时出现问题
            timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
            timesteps -= 1  # 减去1以调整步长
        else:
            # 如果选择的时间步长间隔不被支持，抛出异常
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
    
        # 根据累积 alpha 值计算 sigma 值
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        log_sigmas = np.log(sigmas)  # 计算 sigma 的对数
        # 插值计算 sigma 值以对应时间步长
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
    
        # 如果使用 Karras sigma，则转换 sigma
        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas)
            # 根据 sigma 计算对应的时间步长
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
    
        # 将 sigma 数组与 0.0 连接，并转换为浮点数类型
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
    
        # 将 sigma 和时间步长转换为张量并移动到指定设备
        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        self.timesteps = torch.from_numpy(timesteps).to(device=device)
        self._step_index = None  # 初始化步骤索引为 None
        self._begin_index = None  # 初始化开始索引为 None
        # 将 sigma 移动到 CPU，以避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  
    
        # 初始化导数列表
        self.derivatives = []
    
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep 复制
    # 根据给定的时间步获取对应的索引
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果没有提供时间步调度，则使用类的时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与指定时间步相等的所有索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 对于第一个步骤，选取的 sigma 索引总是第二个索引
        # 如果只有一个索引，则选择最后一个
        # 这样可以确保在从去噪调度的中间开始时不会意外跳过 sigma
        pos = 1 if len(indices) > 1 else 0

        # 返回所选索引的项
        return indices[pos].item()

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制
    # 初始化步骤索引
    def _init_step_index(self, timestep):
        # 如果开始索引为 None，则计算步骤索引
        if self.begin_index is None:
            # 如果时间步是一个张量，则将其转换到相应设备
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 调用 index_for_timestep 方法获取步骤索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 如果有开始索引，则使用它
            self._step_index = self._begin_index

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制
    # 将 sigma 转换为时间 t
    def _sigma_to_t(self, sigma, log_sigmas):
        # 获取 sigma 的对数值，避免对数为负
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # 计算 log_sigma 与 log_sigmas 之间的分布
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 获取 sigma 范围的索引
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        # 获取低和高的 log_sigma 值
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 插值计算 sigma
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # 将插值转换为时间范围
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)  # 确保形状与 sigma 一致
        return t

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras 复制
    # 将 sigma 转换为 Karras 的噪声调度
    def _convert_to_karras(self, in_sigmas: torch.Tensor) -> torch.Tensor:
        """构建 Karras 等人 (2022) 的噪声调度。"""

        # 获取输入 sigma 的最小和最大值
        sigma_min: float = in_sigmas[-1].item()
        sigma_max: float = in_sigmas[0].item()

        # 设置 rho 的值
        rho = 7.0  # 论文中使用的值
        # 创建从 0 到 1 的线性变化
        ramp = np.linspace(0, 1, self.num_inference_steps)
        # 根据 rho 计算最小和最大反 rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 根据反 rho 和线性变化计算 sigmas
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # 定义步骤函数，处理模型输出、时间步和样本
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        order: int = 4,
        return_dict: bool = True,
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise 复制
    # 添加噪声到原始样本
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 的设备和数据类型与 original_samples 相同
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备类型，如果是 mps 并且 timesteps 是浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 否则，将 timesteps 转换为原样本设备的类型
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # 当 scheduler 用于训练时，self.begin_index 为 None，或者管道未实现 set_begin_index
        if self.begin_index is None:
            # 计算每个时间步的索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一个去噪步骤后调用 add_noise（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一个去噪步骤之前调用 add_noise，以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 获取对应时间步的 sigma 值并展平
        sigma = sigmas[step_indices].flatten()
        # 如果 sigma 的维度小于 original_samples，进行维度扩展
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 添加噪声到原样本，生成噪声样本
        noisy_samples = original_samples + noise * sigma
        # 返回生成的噪声样本
        return noisy_samples

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```