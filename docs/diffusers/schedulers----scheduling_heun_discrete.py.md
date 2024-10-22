# `.\diffusers\schedulers\scheduling_heun_discrete.py`

```py
# 版权所有 2024 Katherine Crowson，HuggingFace团队和hlky。保留所有权利。
#
# 根据Apache许可证第2.0版（“许可证”）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，按照许可证分发的软件是按“原样”基础分发，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证具体条款，请参阅许可证。
import math  # 导入数学库以进行数学计算
from typing import List, Optional, Tuple, Union  # 从typing模块导入类型提示

import numpy as np  # 导入NumPy库以处理数组和数学操作
import torch  # 导入PyTorch库用于张量操作

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具中导入混合类和注册函数
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput  # 从调度工具中导入调度相关类

# 从diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 定义扩散时间步数
    max_beta=0.999,  # 定义最大beta值，默认值为0.999
    alpha_transform_type="cosine",  # 定义alpha变换类型，默认值为“cosine”
):
    """
    创建一个beta调度程序，该程序离散化给定的alpha_t_bar函数，该函数定义了
    随着时间的推移（从t = [0,1]）的(1-beta)的累积乘积。

    包含一个alpha_bar函数，该函数接受参数t并将其转换为在扩散过程中
    到该部分的(1-beta)的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 生成的beta数量。
        max_beta (`float`): 使用的最大beta值；使用小于1的值以
                     防止奇点。
        alpha_transform_type (`str`, *可选*, 默认值为`cosine`): alpha_bar的噪声调度类型。
                     从`cosine`或`exp`中选择

    返回：
        betas (`np.ndarray`): 调度器用于调整模型输出的betas
    """
    if alpha_transform_type == "cosine":  # 检查alpha变换类型是否为“cosine”

        def alpha_bar_fn(t):  # 定义alpha_bar函数，接受参数t
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # 计算cosine变换

    elif alpha_transform_type == "exp":  # 检查alpha变换类型是否为“exp”

        def alpha_bar_fn(t):  # 定义alpha_bar函数，接受参数t
            return math.exp(t * -12.0)  # 计算指数变换

    else:  # 如果变换类型不支持
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")  # 抛出错误

    betas = []  # 初始化空列表以存储beta值
    for i in range(num_diffusion_timesteps):  # 遍历每个扩散时间步
        t1 = i / num_diffusion_timesteps  # 计算当前时间步t1
        t2 = (i + 1) / num_diffusion_timesteps  # 计算下一个时间步t2
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))  # 计算beta值并添加到列表中
    return torch.tensor(betas, dtype=torch.float32)  # 返回转换为PyTorch张量的beta列表


class HeunDiscreteScheduler(SchedulerMixin, ConfigMixin):  # 定义Heun离散调度器类，继承自SchedulerMixin和ConfigMixin
    """
    具有Heun步骤的离散beta调度器。

    该模型继承自[`SchedulerMixin`]和[`ConfigMixin`]。查看超类文档以获取库为所有调度器实现的通用
    方法，例如加载和保存。
    # 参数定义部分，描述每个参数的作用和默认值
        Args:
            num_train_timesteps (`int`, defaults to 1000):  # 模型训练的扩散步数
                The number of diffusion steps to train the model.
            beta_start (`float`, defaults to 0.0001):  # 推理的起始 beta 值
                The starting `beta` value of inference.
            beta_end (`float`, defaults to 0.02):  # 推理的最终 beta 值
                The final `beta` value.
            beta_schedule (`str`, defaults to `"linear"`):  # beta 调度策略，映射 beta 范围到一系列 beta
                The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
                `linear` or `scaled_linear`.
            trained_betas (`np.ndarray`, *optional*):  # 直接传递 beta 数组以绕过 beta_start 和 beta_end
                Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
            prediction_type (`str`, defaults to `epsilon`, *optional*):  # 调度函数的预测类型
                Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
                `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
                Video](https://imagen.research.google/video/paper.pdf) paper).
            clip_sample (`bool`, defaults to `True`):  # 为了数值稳定性裁剪预测样本
                Clip the predicted sample for numerical stability.
            clip_sample_range (`float`, defaults to 1.0):  # 样本裁剪的最大幅度，仅在 clip_sample=True 时有效
                The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
            use_karras_sigmas (`bool`, *optional*, defaults to `False`):  # 是否在采样过程中使用 Karras sigma
                Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
                the sigmas are determined according to a sequence of noise levels {σi}.
            timestep_spacing (`str`, defaults to `"linspace"`):  # 时间步的缩放方式
                The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
            steps_offset (`int`, defaults to 0):  # 添加到推理步数的偏移量
                An offset added to the inference steps, as required by some model families.
        """
    
        # 创建与 KarrasDiffusionSchedulers 兼容的名称列表
        _compatibles = [e.name for e in KarrasDiffusionSchedulers]
        # 设置默认顺序为2
        order = 2
    
        # 注册到配置中的构造函数
        @register_to_config
        def __init__(
            self,
            num_train_timesteps: int = 1000,  # 训练的扩散步数，默认1000
            beta_start: float = 0.00085,  # 合理的默认起始 beta 值
            beta_end: float = 0.012,  # 默认的最终 beta 值
            beta_schedule: str = "linear",  # 默认使用线性调度
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 可选的 beta 数组
            prediction_type: str = "epsilon",  # 默认预测类型为 epsilon
            use_karras_sigmas: Optional[bool] = False,  # 默认不使用 Karras sigmas
            clip_sample: Optional[bool] = False,  # 默认不裁剪样本
            clip_sample_range: float = 1.0,  # 默认裁剪范围为1.0
            timestep_spacing: str = "linspace",  # 默认时间步缩放为线性
            steps_offset: int = 0,  # 默认步数偏移为0
    ):
        # 检查是否提供了训练后的贝塔值
        if trained_betas is not None:
            # 将训练后的贝塔值转换为32位浮点张量
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 检查贝塔调度方式是否为线性
        elif beta_schedule == "linear":
            # 生成从beta_start到beta_end的线性贝塔值，数量为num_train_timesteps
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 检查贝塔调度方式是否为缩放线性
        elif beta_schedule == "scaled_linear":
            # 该调度方式特定于潜在扩散模型
            # 生成从beta_start的平方根到beta_end的平方根的线性贝塔值，并平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 检查贝塔调度方式是否为平方余弦（版本2）
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide余弦调度
            # 使用余弦调度生成贝塔值
            self.betas = betas_for_alpha_bar(num_train_timesteps, alpha_transform_type="cosine")
        # 检查贝塔调度方式是否为指数
        elif beta_schedule == "exp":
            # 使用指数调度生成贝塔值
            self.betas = betas_for_alpha_bar(num_train_timesteps, alpha_transform_type="exp")
        # 如果贝塔调度方式未实现，则抛出异常
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算每个时刻的alpha值
        self.alphas = 1.0 - self.betas
        # 计算alpha的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 设置所有时间步的值
        self.set_timesteps(num_train_timesteps, None, num_train_timesteps)
        # 指示是否使用Karras sigma值
        self.use_karras_sigmas = use_karras_sigmas

        # 初始化步索引和开始索引为None
        self._step_index = None
        self._begin_index = None
        # 将sigma移动到CPU以避免过多的CPU/GPU通信
        self.sigmas = self.sigmas.to("cpu")  

    # 从diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep复制
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果未提供调度时间步，则使用当前时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与给定时间步匹配的索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 对于**第一个**步骤，选择的sigma索引总是第二个索引（如果只有一个则为最后一个索引）
        # 确保不会意外跳过一个sigma
        pos = 1 if len(indices) > 1 else 0

        # 返回所选索引的值
        return indices[pos].item()

    @property
    def init_noise_sigma(self):
        # 返回初始噪声分布的标准差
        # 如果时间步间距为线性或尾随，则返回sigma的最大值
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return self.sigmas.max()

        # 否则返回sigma最大值的平方加1的平方根
        return (self.sigmas.max() ** 2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        当前时间步的索引计数器，每次调度器步骤后增加1。
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引，应该通过管道的set_begin_index方法设置。
        """
        return self._begin_index

    # 从diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index复制
    # 设置调度器的开始索引，默认值为0
        def set_begin_index(self, begin_index: int = 0):
            """
            设置调度器的开始索引。此函数应在推理前从管道运行。
    
            参数:
                begin_index (`int`):
                    调度器的开始索引。
            """
            # 将开始索引赋值给实例变量
            self._begin_index = begin_index
    
        # 对模型输入进行缩放，确保与调度器的互换性
        def scale_model_input(
            self,
            sample: torch.Tensor,
            timestep: Union[float, torch.Tensor],
        ) -> torch.Tensor:
            """
            确保与需要根据当前时间步缩放去噪模型输入的调度器互换性。
    
            参数:
                sample (`torch.Tensor`):
                    输入样本。
                timestep (`int`, *可选*):
                    扩散链中的当前时间步。
    
            返回:
                `torch.Tensor`:
                    缩放后的输入样本。
            """
            # 如果步骤索引为空，初始化步骤索引
            if self.step_index is None:
                self._init_step_index(timestep)
    
            # 根据当前步骤索引获取sigma值
            sigma = self.sigmas[self.step_index]
            # 对输入样本进行缩放
            sample = sample / ((sigma**2 + 1) ** 0.5)
            # 返回缩放后的样本
            return sample
    
        # 设置时间步数，包含多个可选参数
        def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
            device: Union[str, torch.device] = None,
            num_train_timesteps: Optional[int] = None,
            timesteps: Optional[List[int]] = None,
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制的
        def _sigma_to_t(self, sigma, log_sigmas):
            # 获取log sigma
            log_sigma = np.log(np.maximum(sigma, 1e-10))
    
            # 获取分布
            dists = log_sigma - log_sigmas[:, np.newaxis]
    
            # 获取sigma范围的低索引
            low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
            # 计算高索引
            high_idx = low_idx + 1
    
            # 获取低和高的log sigma值
            low = log_sigmas[low_idx]
            high = log_sigmas[high_idx]
    
            # 进行sigma的插值
            w = (low - log_sigma) / (low - high)
            # 限制w在0到1之间
            w = np.clip(w, 0, 1)
    
            # 将插值转换为时间范围
            t = (1 - w) * low_idx + w * high_idx
            # 重塑t为与sigma相同的形状
            t = t.reshape(sigma.shape)
            # 返回转换后的时间t
            return t
    
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras 复制的
    # 将输入的 sigma 值转换为 Karras 的噪声调度
        def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
            """构建 Karras et al. (2022) 的噪声调度。"""
    
            # 确保其他调度器复制此函数时不会出错的临时修复
            # TODO: 将此逻辑添加到其他调度器
            if hasattr(self.config, "sigma_min"):
                # 从配置中获取 sigma_min 值
                sigma_min = self.config.sigma_min
            else:
                # 如果没有设置，则 sigma_min 为空
                sigma_min = None
    
            if hasattr(self.config, "sigma_max"):
                # 从配置中获取 sigma_max 值
                sigma_max = self.config.sigma_max
            else:
                # 如果没有设置，则 sigma_max 为空
                sigma_max = None
    
            # 如果 sigma_min 为 None，则取 in_sigmas 的最后一个元素值
            sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
            # 如果 sigma_max 为 None，则取 in_sigmas 的第一个元素值
            sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()
    
            rho = 7.0  # 论文中使用的常量值
            # 创建从 0 到 1 的线性数组，长度为 num_inference_steps
            ramp = np.linspace(0, 1, num_inference_steps)
            # 计算 sigma_min 的倒数
            min_inv_rho = sigma_min ** (1 / rho)
            # 计算 sigma_max 的倒数
            max_inv_rho = sigma_max ** (1 / rho)
            # 根据倒数范围生成新的 sigma 值
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            # 返回生成的 sigma 值
            return sigmas
    
        # 判断是否在一阶状态下
        @property
        def state_in_first_order(self):
            # 如果 dt 为 None，返回 True
            return self.dt is None
    
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制而来
        def _init_step_index(self, timestep):
            # 如果 begin_index 为 None
            if self.begin_index is None:
                # 如果 timestep 是张量，则转换为与 timesteps 相同的设备
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.to(self.timesteps.device)
                # 根据当前时间步获取步骤索引
                self._step_index = self.index_for_timestep(timestep)
            else:
                # 否则使用 _begin_index
                self._step_index = self._begin_index
    
        def step(
            self,
            model_output: Union[torch.Tensor, np.ndarray],
            timestep: Union[float, torch.Tensor],
            sample: Union[torch.Tensor, np.ndarray],
            return_dict: bool = True,
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise 复制而来
        def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    # 返回一个张量，可能用于后续计算
        ) -> torch.Tensor:
            # 确保 sigmas 和 timesteps 与 original_samples 的设备和数据类型一致
            sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
            # 检查设备类型是否为 mps，且 timesteps 为浮点类型
            if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
                # mps 不支持 float64 类型
                schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
                timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
            else:
                # 将 timesteps 转换到 original_samples 的设备
                schedule_timesteps = self.timesteps.to(original_samples.device)
                timesteps = timesteps.to(original_samples.device)
    
            # 当 scheduler 用于训练时，self.begin_index 为 None，或者管道未实现 set_begin_index
            if self.begin_index is None:
                # 根据 timesteps 获取对应的步进索引
                step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
            elif self.step_index is not None:
                # add_noise 在第一次去噪步骤后被调用（用于修补）
                step_indices = [self.step_index] * timesteps.shape[0]
            else:
                # add noise 在第一次去噪步骤之前被调用以创建初始潜在图像（img2img）
                step_indices = [self.begin_index] * timesteps.shape[0]
    
            # 获取对应步进索引的 sigma 值，并将其展平
            sigma = sigmas[step_indices].flatten()
            # 如果 sigma 的维度少于 original_samples，则在最后一维增加维度
            while len(sigma.shape) < len(original_samples.shape):
                sigma = sigma.unsqueeze(-1)
    
            # 添加噪声到原始样本中，生成有噪声的样本
            noisy_samples = original_samples + noise * sigma
            # 返回有噪声的样本
            return noisy_samples
    
        # 返回训练时间步的数量
        def __len__(self):
            return self.config.num_train_timesteps
```