# `.\diffusers\schedulers\scheduling_dpmsolver_multistep_inverse.py`

```py
# 版权声明，说明文件归属及授权信息
# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证进行授权；用户必须遵循许可证使用本文件。
# 可以在以下网址获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，软件按“原样”提供，不附带任何形式的保证或条件。
# 请查看许可证以了解特定的权限和限制。

# 本文件受到 https://github.com/LuChengTHU/dpm-solver 的强烈影响

# 导入数学库
import math
# 导入类型提示
from typing import List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 torch 库
import torch

# 从配置工具导入混合配置类和注册功能
from ..configuration_utils import ConfigMixin, register_to_config
# 导入弃用工具
from ..utils import deprecate
# 导入随机张量工具
from ..utils.torch_utils import randn_tensor
# 导入调度工具
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 设置扩散时间步数
    max_beta=0.999,  # 设置最大 beta 值
    alpha_transform_type="cosine",  # 设置 alpha 转换类型
):
    """
    创建一个 beta 调度，离散化给定的 alpha_t_bar 函数，该函数定义了随时间变化的 (1-beta) 的累积乘积。
    
    包含一个 alpha_bar 函数，该函数接受参数 t 并将其转换为扩散过程的累积乘积。

    参数:
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用低于 1 的值以防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     可选值为 `cosine` 或 `exp`

    返回:
        betas (`np.ndarray`): 调度器用来更新模型输出的 betas。
    """
    # 根据 alpha_transform_type 的类型定义 alpha_bar 函数
    if alpha_transform_type == "cosine":
        # 定义余弦类型的 alpha_bar 函数
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":
        # 定义指数类型的 alpha_bar 函数
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        # 如果 alpha_transform_type 不被支持，抛出异常
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []  # 初始化空列表用于存储 beta 值
    # 遍历每一个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步的 t1 和 t2
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值并添加到列表中，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 返回一个张量形式的 beta 值
    return torch.tensor(betas, dtype=torch.float32)


# DPMSolverMultistepInverseScheduler 类定义，继承调度混合类和配置混合类
class DPMSolverMultistepInverseScheduler(SchedulerMixin, ConfigMixin):
    """
    `DPMSolverMultistepInverseScheduler` 是 [`DPMSolverMultistepScheduler`] 的反向调度器。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关通用的信息，请查看父类文档。
    # 文档字符串，描述库为所有调度程序实现的方法，例如加载和保存功能。
    methods the library implements for all schedulers such as loading and saving.

    """

    # 定义兼容的调度器名称列表，来源于 KarrasDiffusionSchedulers 枚举
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 定义默认的顺序参数
    order = 1

    @register_to_config
    # 初始化方法，接受多个超参数
    def __init__(
        # 训练时间步数，默认值为 1000
        num_train_timesteps: int = 1000,
        # β 值的起始值，默认值为 0.0001
        beta_start: float = 0.0001,
        # β 值的结束值，默认值为 0.02
        beta_end: float = 0.02,
        # β 值的调度方式，默认值为 "linear"
        beta_schedule: str = "linear",
        # 训练后的 β 值，默认为 None，可以是数组或列表
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 求解器的顺序，默认值为 2
        solver_order: int = 2,
        # 预测类型，默认值为 "epsilon"
        prediction_type: str = "epsilon",
        # 是否使用阈值处理，默认值为 False
        thresholding: bool = False,
        # 动态阈值处理的比例，默认值为 0.995
        dynamic_thresholding_ratio: float = 0.995,
        # 采样的最大值，默认值为 1.0
        sample_max_value: float = 1.0,
        # 算法类型，默认值为 "dpmsolver++"
        algorithm_type: str = "dpmsolver++",
        # 求解器类型，默认值为 "midpoint"
        solver_type: str = "midpoint",
        # 最后阶数是否较低，默认值为 True
        lower_order_final: bool = True,
        # 最后一步是否使用欧拉法，默认值为 False
        euler_at_final: bool = False,
        # 是否使用 Karras 的 sigma 值，默认值为 None
        use_karras_sigmas: Optional[bool] = False,
        # λ 最小值裁剪，默认值为负无穷
        lambda_min_clipped: float = -float("inf"),
        # 方差类型，默认值为 None
        variance_type: Optional[str] = None,
        # 时间步间距类型，默认值为 "linspace"
        timestep_spacing: str = "linspace",
        # 步骤偏移量，默认值为 0
        steps_offset: int = 0,
    ):
        # 检查算法类型是否为已弃用类型
        if algorithm_type in ["dpmsolver", "sde-dpmsolver"]:
            # 构建弃用信息消息
            deprecation_message = f"algorithm_type {algorithm_type} is deprecated and will be removed in a future version. Choose from `dpmsolver++` or `sde-dpmsolver++` instead"
            # 调用弃用函数，传递相关信息
            deprecate("algorithm_types dpmsolver and sde-dpmsolver", "1.0.0", deprecation_message)

        # 如果训练的beta值不为None，则初始化self.betas
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果beta_schedule为"linear"，则生成线性beta值
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果beta_schedule为"scaled_linear"，生成特定的beta值
        elif beta_schedule == "scaled_linear":
            # 此调度特定于潜在扩散模型
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 如果beta_schedule为"squaredcos_cap_v2"，生成Glide余弦调度的beta值
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果beta_schedule不在已实现的调度中，则抛出未实现异常
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算alphas值
        self.alphas = 1.0 - self.betas
        # 计算alphas的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 目前只支持VP类型噪声调度
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        # 计算sigma_t值
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        # 计算lambda_t值
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        # 计算sigmas值
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # 设置初始噪声分布的标准差
        self.init_noise_sigma = 1.0

        # DPM-Solver的设置
        if algorithm_type not in ["dpmsolver", "dpmsolver++", "sde-dpmsolver", "sde-dpmsolver++"]:
            # 如果算法类型为"deis"，则注册为"dpmsolver++"
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            # 否则抛出未实现异常
            else:
                raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")

        # 检查solver_type是否合法
        if solver_type not in ["midpoint", "heun"]:
            # 如果solver_type在特定类型中，则注册为"midpoint"
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            # 否则抛出未实现异常
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        # 可设置的值
        self.num_inference_steps = None
        # 创建时间步长的线性数组
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32).copy()
        # 将numpy数组转换为torch张量
        self.timesteps = torch.from_numpy(timesteps)
        # 初始化模型输出列表
        self.model_outputs = [None] * solver_order
        # 初始化低阶数字
        self.lower_order_nums = 0
        # 初始化步骤索引
        self._step_index = None
        # 将sigmas转移到CPU，减少CPU/GPU通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        # 设置是否使用Karras sigmas
        self.use_karras_sigmas = use_karras_sigmas

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加1。
        """
        return self._step_index
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 拷贝而来
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "动态阈值处理：在每个采样步骤中，我们将 s 设置为 xt0（在时间步 t 预测的 x_0）中的某个百分位绝对像素值，
        如果 s > 1，则我们将 xt0 阈值处理到范围 [-s, s]，然后除以 s。动态阈值处理将饱和像素（接近 -1 和 1 的像素）
        向内推，以主动防止每一步的饱和。我们发现动态阈值处理显著提高了照片真实感以及图像-文本对齐，特别是在使用非常大的引导权重时。"
        
        https://arxiv.org/abs/2205.11487
        """
        # 获取输入样本的数据类型
        dtype = sample.dtype
        # 解包输入样本的形状信息，得到批量大小、通道数及其他维度
        batch_size, channels, *remaining_dims = sample.shape
    
        # 如果数据类型不是 float32 或 float64，则将样本转换为 float
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 为了计算分位数而上升精度，并且 CPU 半精度的 clamp 未实现
    
        # 将样本扁平化，以便在每幅图像上进行分位数计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
    
        # 计算样本的绝对值，以获得“某个百分位绝对像素值”
        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"
    
        # 在每个图像的维度上计算绝对样本的分位数
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 限制 s 的范围，最小为 1，最大为配置中的 sample_max_value
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当最小值限制为 1 时，相当于标准剪切到 [-1, 1]
        # 增加维度以便后续广播处理
        s = s.unsqueeze(1)  # (batch_size, 1) 因为 clamp 会在维度 0 上广播
        # 将样本限制在范围 [-s, s] 内，并除以 s
        sample = torch.clamp(sample, -s, s) / s  # "我们将 xt0 阈值处理到范围 [-s, s] 并除以 s"
    
        # 恢复样本的原始形状
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原始数据类型
        sample = sample.to(dtype)
    
        # 返回处理后的样本
        return sample
    
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 拷贝而来
    def _sigma_to_t(self, sigma, log_sigmas):
        # 获取 log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))
    
        # 计算分布
        dists = log_sigma - log_sigmas[:, np.newaxis]
    
        # 获取 sigma 的范围
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
    
        # 获取 low 和 high 的值
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
    
        # 对 sigma 进行插值
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)
    
        # 将插值转换为时间范围
        t = (1 - w) * low_idx + w * high_idx
        # 将 t 形状恢复为 sigma 的形状
        t = t.reshape(sigma.shape)
        # 返回计算得到的 t
        return t
    
    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._sigma_to_alpha_sigma_t 拷贝而来
    def _sigma_to_alpha_sigma_t(self, sigma):
        # 根据 sigma 计算 alpha_t
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        # 计算 sigma_t
        sigma_t = sigma * alpha_t
    
        # 返回 alpha_t 和 sigma_t
        return alpha_t, sigma_t
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras 复制的代码
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构建 Karras 等人（2022年）的噪声调度。"""
    
        # 确保其他复制此函数的调度器不会出错的临时处理
        # TODO: 将此逻辑添加到其他调度器
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min  # 获取配置中的最小 sigma 值
        else:
            sigma_min = None  # 如果没有定义，则设置为 None
    
        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max  # 获取配置中的最大 sigma 值
        else:
            sigma_max = None  # 如果没有定义，则设置为 None
    
        # 如果 sigma_min 为空，则使用输入 sigma 的最后一个值
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        # 如果 sigma_max 为空，则使用输入 sigma 的第一个值
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()
    
        rho = 7.0  # 论文中使用的 rho 值为 7.0
        ramp = np.linspace(0, 1, num_inference_steps)  # 创建线性 ramp 从 0 到 1
        min_inv_rho = sigma_min ** (1 / rho)  # 计算最小 sigma 的倒数
        max_inv_rho = sigma_max ** (1 / rho)  # 计算最大 sigma 的倒数
        # 根据线性 ramp 和倒数值计算 sigmas
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas  # 返回计算得到的 sigmas
    
    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.convert_model_output 复制的代码
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ):
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.dpm_solver_first_order_update 复制的代码
        def dpm_solver_first_order_update(
            self,
            model_output: torch.Tensor,
            *args,
            sample: torch.Tensor = None,
            noise: Optional[torch.Tensor] = None,
            **kwargs,
        ):
    # 返回前一时间步的样本张量
    ) -> torch.Tensor:
            """
            对第一阶 DPMSolver 执行一步（相当于 DDIM）。
    
            参数：
                model_output (`torch.Tensor`):
                    从学习的扩散模型直接输出的张量。
                sample (`torch.Tensor`):
                    扩散过程中生成的当前样本实例。
    
            返回：
                `torch.Tensor`:
                    前一时间步的样本张量。
            """
            # 从位置参数或关键字参数获取当前时间步
            timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
            # 从位置参数或关键字参数获取前一个时间步
            prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
            # 检查样本是否为 None
            if sample is None:
                # 如果存在第三个位置参数，则将其赋值给 sample
                if len(args) > 2:
                    sample = args[2]
                # 如果没有样本，则引发错误
                else:
                    raise ValueError(" missing `sample` as a required keyward argument")
            # 如果当前时间步不为 None
            if timestep is not None:
                # 警告用户时间步已弃用
                deprecate(
                    "timesteps",
                    "1.0.0",
                    "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
                )
    
            # 如果前一个时间步不为 None
            if prev_timestep is not None:
                # 警告用户前一个时间步已弃用
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
            # 根据算法类型计算 x_t
            if self.config.algorithm_type == "dpmsolver++":
                x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
            elif self.config.algorithm_type == "dpmsolver":
                x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
            elif self.config.algorithm_type == "sde-dpmsolver++":
                assert noise is not None  # 确保噪声不为 None
                x_t = (
                    (sigma_t / sigma_s * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            elif self.config.algorithm_type == "sde-dpmsolver":
                assert noise is not None  # 确保噪声不为 None
                x_t = (
                    (alpha_t / alpha_s) * sample
                    - 2.0 * (sigma_t * (torch.exp(h) - 1.0)) * model_output
                    + sigma_t * torch.sqrt(torch.exp(2 * h) - 1.0) * noise
                )
            # 返回计算得到的 x_t
            return x_t
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.multistep_dpm_solver_second_order_update 复制
    # 定义一个多步 DPM 求解器的二阶更新方法
    def multistep_dpm_solver_second_order_update(
            self,
            model_output_list: List[torch.Tensor],  # 模型输出的张量列表
            *args,  # 可变位置参数
            sample: torch.Tensor = None,  # 输入样本，默认为 None
            noise: Optional[torch.Tensor] = None,  # 噪声张量，默认为 None
            **kwargs,  # 可变关键字参数
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler 复制的三阶更新方法
        def multistep_dpm_solver_third_order_update(
            self,
            model_output_list: List[torch.Tensor],  # 模型输出的张量列表
            *args,  # 可变位置参数
            sample: torch.Tensor = None,  # 输入样本，默认为 None
            **kwargs,  # 可变关键字参数
        # 定义初始化步骤索引的方法
        def _init_step_index(self, timestep):
            # 检查 timestep 是否为张量，如果是，则移动到当前时刻的设备
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
    
            # 查找与当前 timestep 匹配的时间索引候选
            index_candidates = (self.timesteps == timestep).nonzero()
    
            # 如果没有找到匹配的索引，则使用最后一个时间索引
            if len(index_candidates) == 0:
                step_index = len(self.timesteps) - 1
            # 如果找到多个匹配的索引，选择第二个索引（确保不会跳过 sigma）
            # 这样可以在去噪调度中间开始时，保证不会意外跳过 sigma
            elif len(index_candidates) > 1:
                step_index = index_candidates[1].item()
            # 如果只找到一个匹配的索引，则使用该索引
            else:
                step_index = index_candidates[0].item()
    
            # 将计算出的步骤索引保存到实例变量中
            self._step_index = step_index
    
        # 定义步骤方法
        def step(
            self,
            model_output: torch.Tensor,  # 模型输出的张量
            timestep: Union[int, torch.Tensor],  # 当前的时间步，整数或张量
            sample: torch.Tensor,  # 输入样本
            generator=None,  # 随机数生成器，默认为 None
            variance_noise: Optional[torch.Tensor] = None,  # 可选的方差噪声张量
            return_dict: bool = True,  # 是否返回字典，默认为 True
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler 复制的模型输入缩放方法
        def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            """
            确保与需要根据当前时间步缩放去噪模型输入的调度器互换性。
    
            参数:
                sample (`torch.Tensor`):
                    输入样本。
    
            返回:
                `torch.Tensor`:
                    一个缩放后的输入样本。
            """
            # 返回原样本，未进行缩放
            return sample
    
        # 定义添加噪声的方法
        def add_noise(
            self,
            original_samples: torch.Tensor,  # 原始样本张量
            noise: torch.Tensor,  # 噪声张量
            timesteps: torch.IntTensor,  # 时间步张量
    ) -> torch.Tensor:  # 定义返回类型为 torch.Tensor 的函数，函数的开始部分
        # 确保 sigmas 和 timesteps 的设备和数据类型与 original_samples 相同
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 如果设备类型为 "mps" 并且 timesteps 为浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            # 将时间步长转换为与 original_samples 相同设备和 float32 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将时间步长转换为与 original_samples 相同设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        step_indices = []  # 初始化一个空列表，用于存储步骤索引
        for timestep in timesteps:  # 遍历每个时间步
            # 找到与当前时间步相等的调度时间步的索引候选
            index_candidates = (schedule_timesteps == timestep).nonzero()
            # 如果没有找到索引候选
            if len(index_candidates) == 0:
                # 使用最后一个调度时间步的索引
                step_index = len(schedule_timesteps) - 1
            # 如果找到多个索引候选
            elif len(index_candidates) > 1:
                # 取第二个候选的索引作为步骤索引
                step_index = index_candidates[1].item()
            else:
                # 只有一个候选，取它的索引
                step_index = index_candidates[0].item()
            # 将步骤索引添加到列表中
            step_indices.append(step_index)

        # 根据步骤索引获取对应的 sigma 值，并将其展平
        sigma = sigmas[step_indices].flatten()
        # 如果 sigma 的维度小于 original_samples 的维度，则在最后增加一个维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 调用 _sigma_to_alpha_sigma_t 方法计算 alpha_t 和 sigma_t
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # 生成带噪声的样本
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        # 返回带噪声的样本
        return noisy_samples

    def __len__(self):  # 定义 __len__ 方法
        # 返回训练时间步的数量
        return self.config.num_train_timesteps
```