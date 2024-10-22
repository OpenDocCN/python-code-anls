# `.\diffusers\schedulers\scheduling_k_dpm_2_discrete.py`

```py
# 版权声明，包含版权年份、作者及其团队信息
# Copyright 2024 Katherine Crowson, The HuggingFace Team and hlky. All rights reserved.
#
# 根据 Apache License 2.0 进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵守许可证的情况下才能使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有约定，软件按照 "AS IS" 原则分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的保证或条件，包括明示或暗示
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参阅许可证以了解具体的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入数学库
import math
# 从 typing 模块导入类型提示
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch

# 从配置实用工具中导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从调度实用工具中导入 KarrasDiffusionSchedulers、SchedulerMixin 和 SchedulerOutput
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 定义扩散时间步数
    max_beta=0.999,  # 定义最大 beta 值，默认为 0.999
    alpha_transform_type="cosine",  # 定义 alpha 转换类型，默认为 'cosine'
):
    """
    创建一个 beta 调度，以离散化给定的 alpha_t_bar 函数，该函数定义了时间 t = [0,1] 的
    (1-beta) 的累积乘积。

    包含一个 alpha_bar 函数，该函数接受一个参数 t，并将其转换为扩散过程的该部分
    (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以防止奇点。
        alpha_transform_type (`str`, *可选*, 默认值为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择

    返回：
        betas (`np.ndarray`): 调度器用于更新模型输出的 betas
    """
    # 如果 alpha 转换类型为 'cosine'
    if alpha_transform_type == "cosine":
        # 定义 alpha_bar 函数，基于余弦函数计算
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 如果 alpha 转换类型为 'exp'
    elif alpha_transform_type == "exp":
        # 定义 alpha_bar 函数，基于指数函数计算
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    # 如果 alpha 转换类型不受支持，抛出错误
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []  # 初始化一个空列表用于存储 beta 值
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps  # 当前时间步
        t2 = (i + 1) / num_diffusion_timesteps  # 下一个时间步
        # 计算 beta 值并添加到列表，限制在最大 beta 范围内
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 返回 beta 值的张量
    return torch.tensor(betas, dtype=torch.float32)


class KDPM2DiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    KDPM2DiscreteScheduler 的灵感来自 DPMSolver2 和论文 
    [Elucidating the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364)。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。查看超类文档以了解库为所有调度器实现的通用方法，如加载和保存。
    """
    # 参数列表说明
        Args:
            num_train_timesteps (`int`, defaults to 1000):  # 模型训练的扩散步骤数，默认为1000
                The number of diffusion steps to train the model.
            beta_start (`float`, defaults to 0.00085):  # 推理的起始 `beta` 值，默认为0.00085
                The starting `beta` value of inference.
            beta_end (`float`, defaults to 0.012):  # 最终的 `beta` 值，默认为0.012
                The final `beta` value.
            beta_schedule (`str`, defaults to `"linear"`):  # beta 的调度方式，默认为线性，可以选择线性或缩放线性
                The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
                `linear` or `scaled_linear`.
            trained_betas (`np.ndarray`, *optional*):  # 可选参数，直接传入 beta 数组以绕过 beta_start 和 beta_end
                Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
            use_karras_sigmas (`bool`, *optional*, defaults to `False`):  # 可选参数，指示是否使用 Karras sigmas 进行噪声调度
                Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
                the sigmas are determined according to a sequence of noise levels {σi}.
            prediction_type (`str`, defaults to `epsilon`, *optional*):  # 预测类型，可选值包括 epsilon、sample 或 v_prediction
                Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
                `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
                Video](https://imagen.research.google/video/paper.pdf) paper).
            timestep_spacing (`str`, defaults to `"linspace"`):  # 时间步长的缩放方式，默认为线性空间
                The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
            steps_offset (`int`, defaults to 0):  # 推理步骤的偏移量，默认为0
                An offset added to the inference steps, as required by some model families.
        """  # 参数说明文档结束
    
        _compatibles = [e.name for e in KarrasDiffusionSchedulers]  # 从 KarrasDiffusionSchedulers 中提取兼容的名称列表
        order = 2  # 设置调度的顺序为2
    
        @register_to_config  # 将此方法注册到配置中
        def __init__(  # 初始化方法
            self,
            num_train_timesteps: int = 1000,  # 默认训练步骤数为1000
            beta_start: float = 0.00085,  # sensible defaults
            beta_end: float = 0.012,  # 默认最终 beta 值
            beta_schedule: str = "linear",  # 默认调度方式为线性
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 可选的训练 beta 数组
            use_karras_sigmas: Optional[bool] = False,  # 默认不使用 Karras sigmas
            prediction_type: str = "epsilon",  # 默认预测类型为 epsilon
            timestep_spacing: str = "linspace",  # 默认时间步长缩放方式为线性空间
            steps_offset: int = 0,  # 默认步骤偏移量为0
    ):
        # 检查是否有训练好的 beta 值
        if trained_betas is not None:
            # 将训练好的 beta 值转换为张量，数据类型为 float32
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 检查 beta 调度是否为线性
        elif beta_schedule == "linear":
            # 生成从 beta_start 到 beta_end 的线性序列，长度为 num_train_timesteps
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 检查 beta 调度是否为缩放线性
        elif beta_schedule == "scaled_linear":
            # 该调度特定于潜在扩散模型
            # 生成从 beta_start 的平方根到 beta_end 的平方根的线性序列，再平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 检查 beta 调度是否为平方余弦
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            # 使用 betas_for_alpha_bar 函数生成 beta 值
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            # 如果 beta 调度不在已实现的范围内，抛出未实现错误
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alpha 值，等于 1 减去 beta 值
        self.alphas = 1.0 - self.betas
        # 计算 alpha 值的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 设置所有时间步的值
        self.set_timesteps(num_train_timesteps, None, num_train_timesteps)

        # 初始化步骤索引和开始索引
        self._step_index = None
        self._begin_index = None
        # 将 sigma 值移动到 CPU，避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self):
        # 返回初始噪声分布的标准差
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            # 返回 sigma 的最大值
            return self.sigmas.max()

        # 返回 sigma 最大值的平方加 1 的平方根
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

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制的
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的开始索引。此函数应在推理之前从管道运行。

        参数：
            begin_index (`int`):
                调度器的开始索引。
        """
        # 设置调度器的开始索引
        self._begin_index = begin_index

    def scale_model_input(
        self,
        # 输入的样本张量
        sample: torch.Tensor,
        # 当前时间步，可以是浮点数或张量
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        确保与需要根据当前时间步调整去噪模型输入的调度器互换性。

        参数：
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                当前扩散链中的时间步。

        返回：
            `torch.Tensor`:
                一个经过缩放的输入样本。
        """
        # 如果步骤索引尚未初始化，则根据时间步初始化它
        if self.step_index is None:
            self._init_step_index(timestep)

        # 根据状态决定使用哪个 sigma 值
        if self.state_in_first_order:
            sigma = self.sigmas[self.step_index]
        else:
            sigma = self.sigmas_interpol[self.step_index]

        # 将输入样本除以 sigma 的平方加一的平方根，进行缩放
        sample = sample / ((sigma**2 + 1) ** 0.5)
        # 返回经过缩放的样本
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        num_train_timesteps: Optional[int] = None,
    @property
    # 判断是否处于一阶状态，即样本是否为 None
    def state_in_first_order(self):
        return self.sample is None

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep 复制而来
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果没有提供调度时间步，则使用默认时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与当前时间步相匹配的索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 对于第一个 `step`，选择第二个索引（或只有一个时选择最后一个索引）
        pos = 1 if len(indices) > 1 else 0

        # 返回对应的索引值
        return indices[pos].item()

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制而来
    def _init_step_index(self, timestep):
        # 如果开始索引为 None，则初始化步骤索引
        if self.begin_index is None:
            # 如果时间步是张量，则转移到相同设备
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 根据时间步索引初始化步骤索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则将步骤索引设置为开始索引
            self._step_index = self._begin_index

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制而来
    def _sigma_to_t(self, sigma, log_sigmas):
        # 计算 sigma 的对数
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # 计算对数 sigma 与给定对数的差异
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 确定 sigma 的范围
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        # 获取低高对数 sigma 值
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 进行 sigma 的插值
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # 将插值转换为时间范围
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        # 返回时间 t
        return t
    # 复制自 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构造 Karras 等人（2022）提出的噪声调度。"""

        # 确保其他调度器复制此函数时不会出现问题的黑客修复
        # TODO: 将此逻辑添加到其他调度器中
        if hasattr(self.config, "sigma_min"):
            # 如果配置中有 sigma_min，则使用它
            sigma_min = self.config.sigma_min
        else:
            # 否则设置为 None
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            # 如果配置中有 sigma_max，则使用它
            sigma_max = self.config.sigma_max
        else:
            # 否则设置为 None
            sigma_max = None

        # 如果 sigma_min 为 None，则使用输入信号的最后一个值
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        # 如果 sigma_max 为 None，则使用输入信号的第一个值
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 论文中使用的值 7.0
        # 生成一个从 0 到 1 的线性 ramp，长度为 num_inference_steps
        ramp = np.linspace(0, 1, num_inference_steps)
        # 计算 sigma_min 的逆 rho 次方
        min_inv_rho = sigma_min ** (1 / rho)
        # 计算 sigma_max 的逆 rho 次方
        max_inv_rho = sigma_max ** (1 / rho)
        # 根据最大和最小的逆值以及 ramp 生成 sigma 序列
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回生成的 sigma 序列
        return sigmas

    def step(
        self,
        model_output: Union[torch.Tensor, np.ndarray],
        timestep: Union[float, torch.Tensor],
        sample: Union[torch.Tensor, np.ndarray],
        return_dict: bool = True,
    # 复制自 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 与 original_samples 具有相同的设备和数据类型
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备是否为 MPS 且 timesteps 是否为浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # MPS 不支持 float64 数据类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            # 将 timesteps 转换为相同设备和 float32 数据类型
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将 schedule_timesteps 转换为 original_samples 设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            # 将 timesteps 转换为 original_samples 设备
            timesteps = timesteps.to(original_samples.device)

        # 当 scheduler 用于训练时，self.begin_index 为 None，或者管道未实现 set_begin_index
        if self.begin_index is None:
            # 根据 timesteps 计算步索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一个去噪步骤后调用 add_noise（用于图像修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一个去噪步骤之前调用 add_noise 以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 根据步索引提取 sigma，并展平为一维
        sigma = sigmas[step_indices].flatten()
        # 如果 sigma 的维度少于 original_samples，则在最后一个维度添加维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 生成带噪声的样本，通过原始样本与噪声和 sigma 的乘积相加
        noisy_samples = original_samples + noise * sigma
        # 返回带噪声的样本
        return noisy_samples

    # 定义 __len__ 方法以返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```