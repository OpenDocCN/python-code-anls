# `.\diffusers\schedulers\scheduling_dpmsolver_sde.py`

```py
# 版权信息，声明版权所有者及许可证信息
# Copyright 2024 Katherine Crowson, The HuggingFace Team and hlky. All rights reserved.
#
# 按照 Apache 许可证第 2.0 版授权
# 该文件只能在遵循许可证的情况下使用
# 可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用的法律另有规定或书面达成协议，否则按 "现状" 分发软件
# 不提供任何形式的担保或条件
# 查看许可证以了解有关权限和限制的具体内容

# 导入数学模块
import math
# 从 typing 导入多个类型
from typing import List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 torch 库
import torch
# 导入 torchsde 库
import torchsde

# 从配置工具模块导入类和方法
from ..configuration_utils import ConfigMixin, register_to_config
# 从调度工具模块导入类
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


class BatchedBrownianTree:
    """封装 torchsde.BrownianTree 以支持批量熵的类。"""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        # 对 t0 和 t1 进行排序，并获取符号信息
        t0, t1, self.sign = self.sort(t0, t1)
        # 从关键字参数中获取 w0，默认初始化为与 x 形状相同的零张量
        w0 = kwargs.get("w0", torch.zeros_like(x))
        # 如果未提供种子，随机生成一个种子
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        # 设置批量处理标志
        self.batched = True
        try:
            # 确保种子的长度与 x 的第一个维度匹配
            assert len(seed) == x.shape[0]
            w0 = w0[0]  # 取第一个 w0
        except TypeError:
            # 如果种子类型不匹配，将其转为列表，并设置为单个种子
            seed = [seed]
            self.batched = False
        # 根据种子创建一组 BrownianTree 实例
        self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        # 返回排序后的值及其符号
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        # 对 t0 和 t1 进行排序
        t0, t1, sign = self.sort(t0, t1)
        # 调用每棵树并将结果堆叠起来，考虑符号
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        # 如果不是批量处理，返回单一结果
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """基于 torchsde.BrownianTree 的噪声采样器。

    参数：
        x (Tensor): 用于生成随机样本的张量，其形状、设备和数据类型将被使用。
        sigma_min (float): 有效区间的下限。
        sigma_max (float): 有效区间的上限。
        seed (int 或 List[int]): 随机种子。如果提供了种子列表而不是单个整数，
            则噪声采样器将为每个批量项目使用一个 BrownianTree，每个都有自己的种子。
        transform (callable): 一个函数，将 sigma 映射到采样器的内部时间步。
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        # 保存变换函数
        self.transform = transform
        # 变换 sigma_min 和 sigma_max，获得 t0 和 t1
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        # 创建 BatchedBrownianTree 实例
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        # 变换 sigma 和 sigma_next，获得 t0 和 t1
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        # 返回计算的噪声，进行归一化
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()
# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的代码
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 生成的 beta 数量
    max_beta=0.999,  # 使用的最大 beta 值，值低于 1 可防止奇点
    alpha_transform_type="cosine",  # alpha_bar 的噪声调度类型，默认为 cosine
):
    """
    创建一个 beta 调度，离散化给定的 alpha_t_bar 函数，该函数定义了时间 t = [0,1] 的
    (1-beta) 的累积乘积。

    包含一个 alpha_bar 函数，该函数接收 t 参数并将其转换为 (1-beta) 的累积乘积
    直至扩散过程的该部分。


    Args:
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用低于 1 的值来防止奇点。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择

    Returns:
        betas (`np.ndarray`): 调度程序用于步骤模型输出的 betas
    """
    # 检查 alpha_transform_type 是否为 cosine
    if alpha_transform_type == "cosine":
        # 定义 alpha_bar_fn 函数，使用 cosine 计算 alpha_bar
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 检查 alpha_transform_type 是否为 exp
    elif alpha_transform_type == "exp":
        # 定义 alpha_bar_fn 函数，使用指数计算 alpha_bar
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    # 抛出不支持的 alpha_transform_type 错误
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []  # 初始化 beta 列表
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps  # 当前时间步
        t2 = (i + 1) / num_diffusion_timesteps  # 下一个时间步
        # 计算 beta 值并添加到 betas 列表，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 返回 beta 的张量表示
    return torch.tensor(betas, dtype=torch.float32)


class DPMSolverSDEScheduler(SchedulerMixin, ConfigMixin):
    """
    DPMSolverSDEScheduler 实现了 [Elucidating the Design Space of Diffusion-Based
    Generative Models](https://huggingface.co/papers/2206.00364) 论文中的随机采样器。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查看超类文档以获取库为所有调度器实现的通用
    方法，例如加载和保存。
    # 定义初始化方法的参数及其默认值
    Args:
        num_train_timesteps (`int`, defaults to 1000):  # 训练模型的扩散步骤数量，默认为1000
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.00085):  # 推理的起始 beta 值，默认为0.00085
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.012):  # 推理的最终 beta 值，默认为0.012
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):  # beta 计划，定义 beta 范围到模型步骤的映射，默认为线性
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, *optional*):  # 可选参数，直接传递 beta 数组以跳过 beta_start 和 beta_end
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):  # 调度函数的预测类型，默认为预测扩散过程的噪声
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):  # 是否在采样过程中使用 Karras sigmas 来调整噪声调度的步长，默认为 False
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        noise_sampler_seed (`int`, *optional*, defaults to `None`):  # 噪声采样器使用的随机种子，默认为 None 时生成随机种子
            The random seed to use for the noise sampler. If `None`, a random seed is generated.
        timestep_spacing (`str`, defaults to `"linspace"`):  # 定义时间步的缩放方式，默认为线性间隔
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):  # 推理步骤的偏移量，某些模型家族所需，默认为0
            An offset added to the inference steps, as required by some model families.
    """

    # 创建一个兼容的调度器列表，包含 KarrasDiffusionSchedulers 中的所有调度器名称
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置调度器的顺序，默认为2
    order = 2

    # 注册到配置中的初始化函数
    @register_to_config
    def __init__(
        # 初始化函数参数及其默认值
        num_train_timesteps: int = 1000,  # 训练模型的扩散步骤数量，默认为1000
        beta_start: float = 0.00085,  # 推理的起始 beta 值，默认为0.00085
        beta_end: float = 0.012,  # 推理的最终 beta 值，默认为0.012
        beta_schedule: str = "linear",  # beta 计划，默认为线性
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 可选参数，跳过 beta_start 和 beta_end
        prediction_type: str = "epsilon",  # 默认预测类型为噪声
        use_karras_sigmas: Optional[bool] = False,  # 是否使用 Karras sigmas，默认为 False
        noise_sampler_seed: Optional[int] = None,  # 噪声采样器的随机种子，默认为 None
        timestep_spacing: str = "linspace",  # 时间步的缩放方式，默认为线性间隔
        steps_offset: int = 0,  # 推理步骤的偏移量，默认为0
    ):
        # 检查训练的 beta 是否为 None
        if trained_betas is not None:
            # 将训练的 beta 转换为张量，数据类型为 float32
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果 beta_schedule 为线性
        elif beta_schedule == "linear":
            # 生成从 beta_start 到 beta_end 的线性间隔值
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果 beta_schedule 为 scaled_linear
        elif beta_schedule == "scaled_linear":
            # 此调度非常特定于潜在扩散模型
            # 生成 beta_start 和 beta_end 的平方根线性间隔值，并平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 如果 beta_schedule 为 squaredcos_cap_v2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果 beta_schedule 不在以上选项中
        else:
            # 抛出未实现的错误
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alphas，等于 1 减去 betas
        self.alphas = 1.0 - self.betas
        # 计算 alphas 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 设置所有值
        self.set_timesteps(num_train_timesteps, None, num_train_timesteps)
        # 记录是否使用 Karras sigma
        self.use_karras_sigmas = use_karras_sigmas
        # 初始化噪声采样器为 None
        self.noise_sampler = None
        # 记录噪声采样器种子
        self.noise_sampler_seed = noise_sampler_seed
        # 初始化步索引为 None
        self._step_index = None
        # 初始化开始索引为 None
        self._begin_index = None
        # 将 sigmas 移动到 CPU，以避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep 复制
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果没有提供 schedule_timesteps，则使用当前的 timesteps
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 查找与 timestep 相等的索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 确保在调度开始时不会意外跳过 sigma
        # 如果 indices 长度大于 1，则 pos 设为 1，否则设为 0
        pos = 1 if len(indices) > 1 else 0

        # 返回相应位置的索引
        return indices[pos].item()

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制
    def _init_step_index(self, timestep):
        # 如果 begin_index 为 None
        if self.begin_index is None:
            # 如果 timestep 是张量，移动到 timesteps 的设备上
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 通过索引方法获取当前步索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 将步索引设为开始索引
            self._step_index = self._begin_index

    @property
    def init_noise_sigma(self):
        # 返回初始噪声分布的标准差
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            # 返回 sigmas 的最大值
            return self.sigmas.max()

        # 计算并返回噪声的标准差
        return (self.sigmas.max() ** 2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度步骤后增加 1。
        """
        # 返回当前步索引
        return self._step_index

    @property
    # 定义获取初始时间步的函数
        def begin_index(self):
            """
            返回初始时间步索引，应该通过 `set_begin_index` 方法设置。
            """
            # 返回当前的初始时间步索引
            return self._begin_index
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler 复制而来，设置初始时间步的函数
        def set_begin_index(self, begin_index: int = 0):
            """
            设置调度器的初始时间步。此函数应在推断前从管道运行。
    
            Args:
                begin_index (`int`):
                    调度器的初始时间步。
            """
            # 将传入的初始时间步索引保存到类属性
            self._begin_index = begin_index
    
        # 定义缩放模型输入的函数
        def scale_model_input(
            self,
            sample: torch.Tensor,
            timestep: Union[float, torch.Tensor],
        ) -> torch.Tensor:
            """
            确保与需要根据当前时间步缩放去噪模型输入的调度器的可互换性。
    
            Args:
                sample (`torch.Tensor`):
                    输入样本。
                timestep (`int`, *optional*):
                    扩散链中的当前时间步。
    
            Returns:
                `torch.Tensor`:
                    缩放后的输入样本。
            """
            # 如果步索引为空，则初始化步索引
            if self.step_index is None:
                self._init_step_index(timestep)
    
            # 获取当前步的 sigma 值
            sigma = self.sigmas[self.step_index]
            # 根据状态选择 sigma 值
            sigma_input = sigma if self.state_in_first_order else self.mid_point_sigma
            # 将样本缩放
            sample = sample / ((sigma_input**2 + 1) ** 0.5)
            # 返回缩放后的样本
            return sample
    
        # 定义设置时间步的函数
        def set_timesteps(
            self,
            num_inference_steps: int,
            device: Union[str, torch.device] = None,
            num_train_timesteps: Optional[int] = None,
        def _second_order_timesteps(self, sigmas, log_sigmas):
            # 定义 sigma 的函数
            def sigma_fn(_t):
                return np.exp(-_t)
    
            # 定义时间的函数
            def t_fn(_sigma):
                return -np.log(_sigma)
    
            # 设置中点比例
            midpoint_ratio = 0.5
            # 获取时间步
            t = t_fn(sigmas)
            # 计算时间间隔
            delta_time = np.diff(t)
            # 提出新时间步
            t_proposed = t[:-1] + delta_time * midpoint_ratio
            # 计算提出的 sigma 值
            sig_proposed = sigma_fn(t_proposed)
            # 将 sigma 转换为时间步
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sig_proposed])
            # 返回时间步数组
            return timesteps
    
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler 复制而来，将 sigma 转换为时间的函数
        def _sigma_to_t(self, sigma, log_sigmas):
            # 获取 sigma 的对数
            log_sigma = np.log(np.maximum(sigma, 1e-10))
    
            # 计算分布
            dists = log_sigma - log_sigmas[:, np.newaxis]
    
            # 获取 sigma 范围的索引
            low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
    
            # 获取低高边界的对数 sigma 值
            low = log_sigmas[low_idx]
            high = log_sigmas[high_idx]
    
            # 对 sigma 进行插值
            w = (low - log_sigma) / (low - high)
            w = np.clip(w, 0, 1)
    
            # 将插值转换为时间范围
            t = (1 - w) * low_idx + w * high_idx
            # 调整形状以匹配输入 sigma 的形状
            t = t.reshape(sigma.shape)
            # 返回时间步
            return t
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras 复制而来
    def _convert_to_karras(self, in_sigmas: torch.Tensor) -> torch.Tensor:
        """构建 Karras 等人（2022）提出的噪声调度。"""
    
        # 获取输入 sigmas 的最小值
        sigma_min: float = in_sigmas[-1].item()
        # 获取输入 sigmas 的最大值
        sigma_max: float = in_sigmas[0].item()
    
        rho = 7.0  # 论文中使用的值
        # 创建从 0 到 1 的线性 ramp
        ramp = np.linspace(0, 1, self.num_inference_steps)
        # 计算最小和最大倒数 sigma
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 计算最终的 sigmas 值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回计算得到的 sigmas
        return sigmas
    
    @property
    def state_in_first_order(self):
        # 判断当前 sample 是否为 None，以确定状态
        return self.sample is None
    
    def step(
        self,
        model_output: Union[torch.Tensor, np.ndarray],
        timestep: Union[float, torch.Tensor],
        sample: Union[torch.Tensor, np.ndarray],
        return_dict: bool = True,
        s_noise: float = 1.0,
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise 复制而来
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 与 original_samples 具有相同的设备和数据类型
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将调度时间步转换为原始样本的设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)
    
        # 当 scheduler 用于训练时，self.begin_index 为 None，或管道未实现 set_begin_index
        if self.begin_index is None:
            # 根据时间步计算步骤索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一个去噪步骤之后调用 add_noise（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一个去噪步骤之前调用 add_noise，以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]
    
        # 根据步骤索引提取 sigma，并展平
        sigma = sigmas[step_indices].flatten()
        # 如果 sigma 的形状小于原始样本，则在最后添加一个维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
    
        # 计算添加噪声后的样本
        noisy_samples = original_samples + noise * sigma
        # 返回添加噪声后的样本
        return noisy_samples
    
    def __len__(self):
        # 返回训练时间步的数量
        return self.config.num_train_timesteps
```