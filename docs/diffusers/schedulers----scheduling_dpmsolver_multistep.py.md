# `.\diffusers\schedulers\scheduling_dpmsolver_multistep.py`

```py
# 版权声明，指明该文件的版权所有者和使用条款
# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# 按照 Apache 2.0 许可证使用本文件的声明
# Licensed under the Apache License, Version 2.0 (the "License");
# 只能在遵守许可证的情况下使用该文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 如果没有法律适用或书面协议的情况下，该文件按 "AS IS" 基础分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解具体的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 免责声明，说明此文件受到特定项目的影响
# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver

# 导入数学模块
import math
# 从 typing 模块导入 List, Optional, Tuple, Union 类型
from typing import List, Optional, Tuple, Union

# 导入 numpy 模块并使用 np 别名
import numpy as np
# 导入 torch 模块
import torch

# 从配置工具中导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从 utils 中导入 deprecate
from ..utils import deprecate
# 从 torch_utils 中导入 randn_tensor
from ..utils.torch_utils import randn_tensor
# 从调度工具中导入 KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


# 从 diffusers 中复制的函数，用于生成 beta 调度
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 指定生成 beta 数量的参数
    max_beta=0.999,  # 设置最大 beta 值的默认参数
    alpha_transform_type="cosine",  # 指定 alpha 转换类型的默认参数
):
    """
    创建一个 beta 调度，它离散化给定的 alpha_t_bar 函数，定义时间 t = [0,1] 上 (1-beta) 的累积乘积。

    包含一个 alpha_bar 函数，该函数接受参数 t 并将其转换为该部分扩散过程的 (1-beta) 的累积乘积。

    参数:
        num_diffusion_timesteps (`int`): 生成 beta 的数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择

    返回:
        betas (`np.ndarray`): 调度器用于更新模型输出的 betas
    """
    # 检查 alpha_transform_type 是否为 "cosine"
    if alpha_transform_type == "cosine":
        # 定义 alpha_bar_fn 函数，计算 cos 函数的平方
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 检查 alpha_transform_type 是否为 "exp"
    elif alpha_transform_type == "exp":
        # 定义 alpha_bar_fn 函数，计算指数函数
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    # 如果 alpha_transform_type 不是支持的类型，则引发错误
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化 betas 列表
    betas = []
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值并添加到 betas 列表中，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 betas 列表转换为张量并返回
    return torch.tensor(betas, dtype=torch.float32)


# 从 diffusers 中复制的函数，用于重新调整 beta
def rescale_zero_terminal_snr(betas):
    """
    根据 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 重新调整 beta 以具有零终端 SNR

    参数:
        betas (`torch.Tensor`):
            用于初始化调度器的 beta。
    # 返回 rescaled betas，且终端信噪比为零
    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算累积乘积的平方根

    # 存储旧值
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 克隆第一个 alphas_bar_sqrt 的值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 克隆最后一个 alphas_bar_sqrt 的值

    # 平移，使最后一个时间步为零
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 从每个元素中减去最后一个值

    # 缩放，使第一个时间步恢复到旧值
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 缩放 alphas_bar_sqrt

    # 将 alphas_bar_sqrt 转换为 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方根
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 还原累积乘积
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个 alphas_bar 的值添加到 alphas 前面
    betas = 1 - alphas  # 计算 betas

    # 返回计算后的 betas
    return betas
# 定义一个多步调度器类，专用于快速高阶求解扩散常微分方程
class DPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `DPMSolverMultistepScheduler` 是一个快速的专用高阶求解器，用于扩散 ODE。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查看父类文档以了解该库为所有调度器实现的通用方法，例如加载和保存。
    """

    # 存储与 KarrasDiffusionSchedulers 兼容的调度器名称列表
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置默认的求解器阶数为 1
    order = 1

    @register_to_config
    def __init__(
        # 训练的时间步数，默认为 1000
        num_train_timesteps: int = 1000,
        # 初始 beta 值，默认为 0.0001
        beta_start: float = 0.0001,
        # 最终 beta 值，默认为 0.02
        beta_end: float = 0.02,
        # beta 的调度类型，默认为线性
        beta_schedule: str = "linear",
        # 经过训练的 beta 值，默认为 None
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 求解器阶数，默认为 2
        solver_order: int = 2,
        # 预测类型，默认为 "epsilon"
        prediction_type: str = "epsilon",
        # 是否启用阈值处理，默认为 False
        thresholding: bool = False,
        # 动态阈值处理比例，默认为 0.995
        dynamic_thresholding_ratio: float = 0.995,
        # 样本的最大值，默认为 1.0
        sample_max_value: float = 1.0,
        # 算法类型，默认为 "dpmsolver++"
        algorithm_type: str = "dpmsolver++",
        # 求解器类型，默认为 "midpoint"
        solver_type: str = "midpoint",
        # 是否在最后阶段使用低阶方法，默认为 True
        lower_order_final: bool = True,
        # 最后阶段是否使用欧拉法，默认为 False
        euler_at_final: bool = False,
        # 是否使用 Karras 的 sigma 值，默认为 None
        use_karras_sigmas: Optional[bool] = False,
        # 是否使用 LU 的 lambda 值，默认为 None
        use_lu_lambdas: Optional[bool] = False,
        # 最终 sigma 类型，默认为 "zero"
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
        # 最小 lambda 值被裁剪，默认为 -inf
        lambda_min_clipped: float = -float("inf"),
        # 方差类型，默认为 None
        variance_type: Optional[str] = None,
        # 时间步的间隔类型，默认为 "linspace"
        timestep_spacing: str = "linspace",
        # 步骤偏移量，默认为 0
        steps_offset: int = 0,
        # 是否在 SNR 为零时重新缩放 beta，默认为 False
        rescale_betas_zero_snr: bool = False,
    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后会增加 1。
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过 `set_begin_index` 方法从管道设置。
        """
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的起始索引。该函数应在推理前通过管道运行。

        参数：
            begin_index (`int`):
                调度器的起始索引。
        """
        # 更新调度器的起始索引
        self._begin_index = begin_index

    def set_timesteps(
        # 设置推理步骤的数量，默认为 None
        num_inference_steps: int = None,
        # 设备类型，可以是字符串或 torch.device，默认为 None
        device: Union[str, torch.device] = None,
        # 指定时间步，默认为 None
        timesteps: Optional[List[int]] = None,
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 中复制
    # 定义一个私有方法，进行动态阈值采样，输入为一个张量，返回处理后的张量
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "动态阈值处理：在每个采样步骤中，我们将 s 设置为 xt0（在时间步 t 对 x_0 的预测）中的某个百分位绝对像素值，如果 s > 1，则将 xt0 阈值化到范围 [-s, s]，然后除以 s。动态阈值处理将饱和像素（接近 -1 和 1 的像素）推入内部，从而在每一步主动防止像素饱和。我们发现动态阈值处理可以显著提高照片真实感以及图像与文本的对齐，尤其是在使用非常大的引导权重时。"

        https://arxiv.org/abs/2205.11487
        """
        # 获取输入样本的数值类型
        dtype = sample.dtype
        # 解包样本的形状，获取批量大小、通道数及其余维度
        batch_size, channels, *remaining_dims = sample.shape

        # 如果数据类型不是 float32 或 float64，则将样本上升为 float 类型
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 为量化计算上升数据类型，且 cpu half 不支持 clamp

        # 将样本展平，以便对每幅图像进行量化计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        # 计算样本的绝对值，用于获取"某个百分位绝对像素值"
        abs_sample = sample.abs()  # "某个百分位绝对像素值"

        # 计算绝对样本的指定百分位值
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 将 s 的值限制在 [1, sample_max_value] 范围内
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当 clamped 到 min=1 时，相当于标准的裁剪到 [-1, 1]
        # 在第一维增加一个维度，以便后续操作中能够正确广播
        s = s.unsqueeze(1)  # (batch_size, 1) 因为 clamp 将在 dim=0 上广播
        # 将样本的值限制在 [-s, s] 范围内，并将其归一化
        sample = torch.clamp(sample, -s, s) / s  # "我们将 xt0 阈值化到范围 [-s, s]，然后除以 s"

        # 恢复样本的原始形状
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原始的数据类型
        sample = sample.to(dtype)

        # 返回处理后的样本
        return sample

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 拷贝的方法
    def _sigma_to_t(self, sigma, log_sigmas):
        # 计算 log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # 计算分布
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 获取 sigma 的范围
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        # 获取低和高的 log sigma 值
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 进行 sigma 的插值
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # 将插值转换为时间范围
        t = (1 - w) * low_idx + w * high_idx
        # 将 t 的形状恢复为 sigma 的形状
        t = t.reshape(sigma.shape)
        # 返回时间值
        return t

    # 定义一个方法，将 sigma 转换为 alpha_t 和 sigma_t
    def _sigma_to_alpha_sigma_t(self, sigma):
        # 计算 alpha_t，使用 sigma 的平方根
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        # 计算 sigma_t，通过 sigma 和 alpha_t 进行计算
        sigma_t = sigma * alpha_t

        # 返回 alpha_t 和 sigma_t
        return alpha_t, sigma_t

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras 拷贝的方法
    # 定义一个私有方法，用于将输入的 sigma 值转换为 Karras 的噪声调度
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构建 Karras 等人 (2022) 的噪声调度。"""

        # 确保其他调度器复制此函数时不会出错的 Hack
        # TODO: 将此逻辑添加到其他调度器中
        # 检查配置中是否存在 sigma_min 属性
        if hasattr(self.config, "sigma_min"):
            # 如果存在，则获取其值
            sigma_min = self.config.sigma_min
        else:
            # 否则，将 sigma_min 设置为 None
            sigma_min = None

        # 检查配置中是否存在 sigma_max 属性
        if hasattr(self.config, "sigma_max"):
            # 如果存在，则获取其值
            sigma_max = self.config.sigma_max
        else:
            # 否则，将 sigma_max 设置为 None
            sigma_max = None

        # 如果 sigma_min 为 None，则使用 in_sigmas 中最后一个值
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        # 如果 sigma_max 为 None，则使用 in_sigmas 中第一个值
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        # rho 的值为论文中使用的 7.0
        rho = 7.0  # 7.0 是论文中使用的值
        # 创建一个从 0 到 1 的线性 ramp，长度为 num_inference_steps
        ramp = np.linspace(0, 1, num_inference_steps)
        # 计算 sigma_min 的倒数 rho
        min_inv_rho = sigma_min ** (1 / rho)
        # 计算 sigma_max 的倒数 rho
        max_inv_rho = sigma_max ** (1 / rho)
        # 根据 ramp 计算出对应的 sigma 值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回计算得到的 sigma 值
        return sigmas

    # 定义一个私有方法，用于将输入的 lambda 值转换为 Lu 的噪声调度
    def _convert_to_lu(self, in_lambdas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构建 Lu 等人 (2022) 的噪声调度。"""

        # 获取输入 lambdas 的最小值（最后一个值）
        lambda_min: float = in_lambdas[-1].item()
        # 获取输入 lambdas 的最大值（第一个值）
        lambda_max: float = in_lambdas[0].item()

        # rho 的值为论文中使用的 1.0
        rho = 1.0  # 1.0 是论文中使用的值
        # 创建一个从 0 到 1 的线性 ramp，长度为 num_inference_steps
        ramp = np.linspace(0, 1, num_inference_steps)
        # 计算 lambda_min 的倒数 rho
        min_inv_rho = lambda_min ** (1 / rho)
        # 计算 lambda_max 的倒数 rho
        max_inv_rho = lambda_max ** (1 / rho)
        # 根据 ramp 计算出对应的 lambda 值
        lambdas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回计算得到的 lambda 值
        return lambdas

    # 定义一个方法，用于转换模型输出
    def convert_model_output(
        self,
        model_output: torch.Tensor,  # 模型的输出张量
        *args,  # 额外的位置参数
        sample: torch.Tensor = None,  # 可选的样本张量
        **kwargs,  # 额外的关键字参数
    # 定义一个方法，用于 DPM 求解器的一阶更新
    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,  # 模型的输出张量
        *args,  # 额外的位置参数
        sample: torch.Tensor = None,  # 可选的样本张量
        noise: Optional[torch.Tensor] = None,  # 可选的噪声张量
        **kwargs,  # 额外的关键字参数
    # 返回一个张量，表示第一阶 DPMSolver 的一步（等效于 DDIM）
    ) -> torch.Tensor:
            """
            一步用于第一阶 DPMSolver（等效于 DDIM）。
            
            参数：
                model_output (`torch.Tensor`):
                    从学习的扩散模型直接输出的张量。
                sample (`torch.Tensor`):
                    扩散过程中创建的当前样本实例。
            
            返回：
                `torch.Tensor`:
                    上一个时间步的样本张量。
            """
            # 从参数中获取当前时间步，若没有则从关键字参数中获取
            timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
            # 从参数中获取上一个时间步，若没有则从关键字参数中获取
            prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
            # 如果样本为空，尝试从参数中获取样本
            if sample is None:
                if len(args) > 2:
                    sample = args[2]
                else:
                    # 抛出错误，样本为必需的关键字参数
                    raise ValueError(" missing `sample` as a required keyward argument")
            # 如果时间步不为空，发出弃用警告
            if timestep is not None:
                deprecate(
                    "timesteps",
                    "1.0.0",
                    "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
                )
            
            # 如果上一个时间步不为空，发出弃用警告
            if prev_timestep is not None:
                deprecate(
                    "prev_timestep",
                    "1.0.0",
                    "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
                )
    
            # 获取当前和前一个时间步的 sigma 值
            sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
            # 将 sigma 转换为 alpha 和 sigma_t
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
            # 将前一个 sigma 转换为 alpha 和 sigma_s
            alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
            # 计算 lambda_t 和 lambda_s
            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
            lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    
            # 计算 h 值
            h = lambda_t - lambda_s
            # 根据配置的算法类型进行不同的计算
            if self.config.algorithm_type == "dpmsolver++":
                x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
            elif self.config.algorithm_type == "dpmsolver":
                x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
            elif self.config.algorithm_type == "sde-dpmsolver++":
                assert noise is not None  # 确保噪声不为空
                x_t = (
                    (sigma_t / sigma_s * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.config.algorithm_type == "sde-dpmsolver":
                assert noise is not None  # 确保噪声不为空
                x_t = (
                    (alpha_t / alpha_s) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            # 返回计算得到的样本张量
            return x_t
    
        # 定义多步 DPM 求解器的二阶更新方法
        def multistep_dpm_solver_second_order_update(
            self,
            model_output_list: List[torch.Tensor],  # 模型输出列表
            *args,  # 额外的参数
            sample: torch.Tensor = None,  # 当前样本
            noise: Optional[torch.Tensor] = None,  # 可选噪声
            **kwargs,  # 额外的关键字参数
    # 定义一个三阶更新的多步 DPM 求解器方法
    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[torch.Tensor],  # 输入的模型输出列表，包含多个张量
        *args,  # 其他位置参数
        sample: torch.Tensor = None,  # 输入样本，默认为 None
        **kwargs,  # 其他关键字参数
    # 定义一个用于获取时间步索引的方法
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果未提供调度时间步，则使用默认时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 查找与给定时间步相等的调度时间步的索引
        index_candidates = (schedule_timesteps == timestep).nonzero()

        # 如果没有找到匹配的索引
        if len(index_candidates) == 0:
            # 设置步索引为时间步列表的最后一个索引
            step_index = len(self.timesteps) - 1
        # 对于多个匹配的情况，选择第二个索引（或最后一个索引）
        # 以确保不会意外跳过 sigma
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            # 如果只找到一个匹配，取第一个索引
            step_index = index_candidates[0].item()

        # 返回计算得到的步索引
        return step_index

    # 初始化调度器的步索引计数器的方法
    def _init_step_index(self, timestep):
        """
        初始化调度器的步索引计数器。
        """

        # 如果开始索引为 None
        if self.begin_index is None:
            # 如果时间步是张量类型，则将其转移到时间步的设备上
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 根据时间步获取步索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 如果开始索引已定义，则直接使用它
            self._step_index = self._begin_index

    # 定义一个步骤方法，执行一个更新步骤
    def step(
        self,
        model_output: torch.Tensor,  # 模型输出的张量
        timestep: Union[int, torch.Tensor],  # 当前的时间步，可以是整数或张量
        sample: torch.Tensor,  # 输入样本
        generator=None,  # 可选的生成器
        variance_noise: Optional[torch.Tensor] = None,  # 可选的方差噪声张量
        return_dict: bool = True,  # 是否返回字典格式的结果
    # 定义一个缩放模型输入的方法
    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器互换。

        参数：
            sample (`torch.Tensor`):
                输入样本。

        返回：
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 返回输入样本，当前没有进行缩放
        return sample

    # 定义一个添加噪声的方法
    def add_noise(
        self,
        original_samples: torch.Tensor,  # 原始样本的张量
        noise: torch.Tensor,  # 要添加的噪声张量
        timesteps: torch.IntTensor,  # 时间步的整数张量
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 与 original_samples 在同一设备和数据类型上
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备是否为 mps 且 timesteps 是否为浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将 timesteps 转换到与 original_samples 相同的设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # 当 scheduler 用于训练或 pipeline 未实现 set_begin_index 时，begin_index 为 None
        if self.begin_index is None:
            # 根据 timesteps 计算 step_indices
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一次去噪步骤后调用 add_noise（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一次去噪步骤之前调用 add_noise 以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 根据 step_indices 获取对应的 sigmas，并将其展平
        sigma = sigmas[step_indices].flatten()
        # 通过增加维度来匹配 sigma 的形状与 original_samples 的形状
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 将 sigma 转换为 alpha_t 和 sigma_t
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # 生成带噪声的样本
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        # 返回带噪声的样本
        return noisy_samples

    def __len__(self):
        # 返回训练时间步的数量
        return self.config.num_train_timesteps
```