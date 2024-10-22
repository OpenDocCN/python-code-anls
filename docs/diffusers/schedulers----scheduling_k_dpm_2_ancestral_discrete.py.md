# `.\diffusers\schedulers\scheduling_k_dpm_2_ancestral_discrete.py`

```py
# 版权声明，声明文件的版权归Katherine Crowson、HuggingFace团队及hlky所有
# 
# 根据Apache许可证第2.0版（“许可证”）进行许可；
# 除非遵循许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“原样”基础提供的，
# 不提供任何形式的明示或暗示的担保或条件。
# 请参见许可证以获取有关权限和限制的具体规定。

# 导入数学库
import math
# 从类型提示模块导入List、Optional、Tuple和Union
from typing import List, Optional, Tuple, Union

# 导入numpy库并简化为np
import numpy as np
# 导入torch库
import torch

# 从配置工具模块导入ConfigMixin和register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从torch工具模块导入randn_tensor函数
from ..utils.torch_utils import randn_tensor
# 从调度工具模块导入KarrasDiffusionSchedulers、SchedulerMixin和SchedulerOutput
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


# 从diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 定义要生成的beta数量
    max_beta=0.999,  # 最大beta值，默认为0.999
    alpha_transform_type="cosine",  # alpha变换类型，默认为“cosine”
):
    """
    创建一个beta调度，离散化给定的alpha_t_bar函数，该函数定义了
    (1-beta)随时间的累积乘积，从t=[0,1]。

    包含一个函数alpha_bar，它接受一个参数t并将其转换为(1-beta)的累积乘积
    到扩散过程的该部分。

    参数：
        num_diffusion_timesteps (`int`): 要生成的beta数量。
        max_beta (`float`): 要使用的最大beta；使用小于1的值以
                     防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为`cosine`): alpha_bar的噪声调度类型。
                     从`cosine`或`exp`中选择。

    返回：
        betas (`np.ndarray`): 调度器用于模型输出的betas
    """
    # 如果alpha变换类型为“cosine”
    if alpha_transform_type == "cosine":
        # 定义alpha_bar_fn函数，接受参数t
        def alpha_bar_fn(t):
            # 返回经过余弦函数转换的值
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 如果alpha变换类型为“exp”
    elif alpha_transform_type == "exp":
        # 定义alpha_bar_fn函数，接受参数t
        def alpha_bar_fn(t):
            # 返回指数衰减的值
            return math.exp(t * -12.0)

    # 如果alpha变换类型不受支持，抛出异常
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []  # 初始化一个空列表，用于存储beta值
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps  # 计算当前时间步的比例
        t2 = (i + 1) / num_diffusion_timesteps  # 计算下一个时间步的比例
        # 计算beta值，并限制在max_beta范围内
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 返回转换为torch张量的beta值，数据类型为float32
    return torch.tensor(betas, dtype=torch.float32)


# 定义KDPM2AncestralDiscreteScheduler类，继承自SchedulerMixin和ConfigMixin
class KDPM2AncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    KDPM2DiscreteScheduler与祖先采样，灵感来自于DPMSolver2和[Elucidating
    the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364)论文中的算法2。

    此模型继承自[`SchedulerMixin`]和[`ConfigMixin`]。请查阅超类文档以获取通用
    # 方法库为所有调度器实现的功能，如加载和保存。

    # 参数说明：
    # num_train_timesteps（`int`，默认为1000）：
    # 训练模型的扩散步骤数。
    # beta_start（`float`，默认为0.00085）：
    # 推理的起始`beta`值。
    # beta_end（`float`，默认为0.012）：
    # 最终的`beta`值。
    # beta_schedule（`str`，默认为`"linear"`）：
    # beta计划，将beta范围映射到一系列用于模型步进的betas。可选值为
    # `linear`或`scaled_linear`。
    # trained_betas（`np.ndarray`，*可选*）：
    # 直接传递betas数组到构造函数，以绕过`beta_start`和`beta_end`。
    # use_karras_sigmas（`bool`，*可选*，默认为`False`）：
    # 是否在采样过程中使用Karras sigmas作为噪声调度中的步长。如果为`True`，
    # sigmas根据一系列噪声水平{σi}确定。
    # prediction_type（`str`，默认为`epsilon`，*可选*）：
    # 调度器函数的预测类型；可以是`epsilon`（预测扩散过程的噪声），
    # `sample`（直接预测有噪声的样本）或`v_prediction`（参见[Imagen Video](https://imagen.research.google/video/paper.pdf)论文第2.4节）。
    # timestep_spacing（`str`，默认为`"linspace"`）：
    # 时间步的缩放方式。有关更多信息，请参考[Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)的表2。
    # steps_offset（`int`，默认为0）：
    # 添加到推理步骤的偏移量，某些模型系列所需。

    # 兼容的调度器列表，提取KarrasDiffusionSchedulers的名称
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置阶数为2
    order = 2

    # 注册到配置中的初始化方法
    @register_to_config
    def __init__(
        # 定义num_train_timesteps参数，默认为1000
        num_train_timesteps: int = 1000,
        # 定义beta_start参数，默认为0.00085，设置合理的默认值
        beta_start: float = 0.00085,  # sensible defaults
        # 定义beta_end参数，默认为0.012
        beta_end: float = 0.012,
        # 定义beta_schedule参数，默认为"linear"
        beta_schedule: str = "linear",
        # 定义trained_betas参数，默认为None，类型为np.ndarray或float列表
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 定义use_karras_sigmas参数，默认为False
        use_karras_sigmas: Optional[bool] = False,
        # 定义prediction_type参数，默认为"epsilon"
        prediction_type: str = "epsilon",
        # 定义timestep_spacing参数，默认为"linspace"
        timestep_spacing: str = "linspace",
        # 定义steps_offset参数，默认为0
        steps_offset: int = 0,
    ):
        # 检查是否提供了训练好的 beta 值
        if trained_betas is not None:
            # 如果提供了，使用这些 beta 值创建一个张量
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 检查 beta 调度类型是否为线性
        elif beta_schedule == "linear":
            # 创建一个从 beta_start 到 beta_end 的线性间隔张量
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 检查 beta 调度类型是否为缩放线性
        elif beta_schedule == "scaled_linear":
            # 该调度非常特定于潜在扩散模型
            # 创建从 beta_start 的平方根到 beta_end 的平方根的线性间隔，然后平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 检查 beta 调度类型是否为平方余弦调度 v2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            # 使用指定的函数为 alpha_bar 计算 beta 值
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            # 如果没有匹配的调度类型，抛出未实现的错误
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alpha 值，alpha 等于 1 减去 beta
        self.alphas = 1.0 - self.betas
        # 计算 alpha 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 设置所有时间步的值
        self.set_timesteps(num_train_timesteps, None, num_train_timesteps)
        # 初始化步骤索引为 None
        self._step_index = None
        # 初始化开始索引为 None
        self._begin_index = None
        # 将 sigmas 张量移动到 CPU，以减少 CPU/GPU 之间的通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self):
        # 返回初始噪声分布的标准差
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            # 如果时间步间隔为线性或拖尾，返回 sigmas 的最大值
            return self.sigmas.max()

        # 否则，计算并返回基于 sigmas 最大值的结果
        return (self.sigmas.max() ** 2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加 1。
        """
        # 返回当前的步骤索引
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过 `set_begin_index` 方法从管道设置。
        """
        # 返回开始索引
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制而来
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的开始索引。该函数应在推理之前从管道运行。

        Args:
            begin_index (`int`):
                调度器的开始索引。
        """
        # 设置开始索引
        self._begin_index = begin_index

    def scale_model_input(
        self,
        # 输入的样本张量
        sample: torch.Tensor,
        # 当前时间步，可以是浮点数或张量
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器互换性。

        参数:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                当前扩散链中的时间步。

        返回:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 如果 step_index 未初始化，调用初始化函数
        if self.step_index is None:
            self._init_step_index(timestep)

        # 如果当前状态为一阶状态，获取当前步的 sigma
        if self.state_in_first_order:
            sigma = self.sigmas[self.step_index]
        else:
            # 否则，获取插值后的 sigma
            sigma = self.sigmas_interpol[self.step_index - 1]

        # 对样本进行缩放，缩放因子为 sqrt(sigma^2 + 1)
        sample = sample / ((sigma**2 + 1) ** 0.5)
        # 返回缩放后的样本
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = None,
        num_train_timesteps: Optional[int] = None,
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制的函数
    def _sigma_to_t(self, sigma, log_sigmas):
        # 获取 sigma 的对数值
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # 获取分布
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 获取 sigma 的范围
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        # 获取低和高的对数 sigma
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 插值 sigma
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # 将插值转换为时间范围
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        # 返回时间值
        return t

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras 复制的函数
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构建 Karras et al. (2022) 的噪声调度。"""

        # 确保其他复制此函数的调度器不会出错的 hack
        # TODO: 将此逻辑添加到其他调度器中
        if hasattr(self.config, "sigma_min"):
            # 如果配置中有 sigma_min，使用其值
            sigma_min = self.config.sigma_min
        else:
            # 否则设为 None
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            # 如果配置中有 sigma_max，使用其值
            sigma_max = self.config.sigma_max
        else:
            # 否则设为 None
            sigma_max = None

        # 如果 sigma_min 仍为 None，使用输入 sigma 的最后一个值
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        # 如果 sigma_max 仍为 None，使用输入 sigma 的第一个值
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 论文中使用的值
        # 创建从 0 到 1 的线性空间
        ramp = np.linspace(0, 1, num_inference_steps)
        # 计算 sigma 的逆
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 计算新的 sigma 值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回新的 sigma 值
        return sigmas

    @property
    def state_in_first_order(self):
        # 返回 sample 是否为 None，判断是否处于一阶状态
        return self.sample is None
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep 复制而来
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果未提供调度时间步，则使用实例的时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
    
        # 找到与给定时间步匹配的索引
        indices = (schedule_timesteps == timestep).nonzero()
    
        # 对于**第一个**步骤，sigma 索引始终是第二个索引
        # 如果只有一个索引，则使用最后一个索引
        pos = 1 if len(indices) > 1 else 0
    
        # 返回对应的索引值
        return indices[pos].item()
    
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制而来
    def _init_step_index(self, timestep):
        # 如果开始索引为空
        if self.begin_index is None:
            # 如果时间步是张量，将其移动到正确的设备
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 初始化步骤索引为对应的时间步索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则使用已定义的开始索引
            self._step_index = self._begin_index
    
    def step(
        # 模型输出，类型可以是 torch.Tensor 或 np.ndarray
        model_output: Union[torch.Tensor, np.ndarray],
        # 当前时间步，可以是浮点数或张量
        timestep: Union[float, torch.Tensor],
        # 输入样本，可以是张量或数组
        sample: Union[torch.Tensor, np.ndarray],
        # 可选的随机生成器
        generator: Optional[torch.Generator] = None,
        # 是否返回字典格式的结果，默认为 True
        return_dict: bool = True,
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise 复制而来
    def add_noise(
        # 原始样本，类型为 torch.Tensor
        original_samples: torch.Tensor,
        # 添加的噪声，类型为 torch.Tensor
        noise: torch.Tensor,
        # 时间步，类型为 torch.Tensor
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 与 original_samples 具有相同的设备和数据类型
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 如果设备类型是 "mps" 且 timesteps 是浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            # 将 timesteps 转换为与 original_samples 相同的设备和 float32 类型
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将 schedule_timesteps 转换为与 original_samples 相同的设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            # 将 timesteps 转换为与 original_samples 相同的设备
            timesteps = timesteps.to(original_samples.device)

        # 当 scheduler 用于训练时 self.begin_index 为 None，或管道未实现 set_begin_index
        if self.begin_index is None:
            # 根据 schedule_timesteps 计算每个 timesteps 的索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一个去噪步骤之后调用 add_noise（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一个去噪步骤之前调用 add_noise 以创建初始潜在图像 (img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 根据 step_indices 获取对应的 sigma 值，并展平为一维
        sigma = sigmas[step_indices].flatten()
        # 在 sigma 的维度小于 original_samples 的维度时，扩展 sigma 维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 生成带噪声的样本，原始样本加上噪声和 sigma 的乘积
        noisy_samples = original_samples + noise * sigma
        # 返回生成的带噪声样本
        return noisy_samples

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```