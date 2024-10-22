# `.\diffusers\schedulers\scheduling_euler_discrete.py`

```py
# 版权所有 2024 Katherine Crowson 和 The HuggingFace 团队。所有权利保留。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行许可；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有规定，分发在许可证下的软件均按“原样”提供，
# 不附带任何形式的担保或条件，无论是明示还是暗示。
# 请参见许可证以获取特定语言的权限和
# 限制条款。

import math  # 导入数学模块以进行数学运算
from dataclasses import dataclass  # 从数据类模块导入dataclass装饰器
from typing import List, Optional, Tuple, Union  # 导入类型注解，用于类型检查和文档

import numpy as np  # 导入NumPy库以进行数值计算
import torch  # 导入PyTorch库以进行深度学习操作

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入混合配置和注册功能
from ..utils import BaseOutput, logging  # 从实用工具导入基本输出类和日志记录功能
from ..utils.torch_utils import randn_tensor  # 从PyTorch工具导入生成随机张量的功能
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin  # 从调度工具导入调度器类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例，名称为当前模块名

@dataclass
# 从 diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput 复制，DDPM 转换为 EulerDiscrete
class EulerDiscreteSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像):
            先前时间步的计算样本 `(x_{t-1})`。 `prev_sample` 应作为下一个模型输入用于
            去噪循环。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像):
            基于当前时间步模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进展或指导。
    """

    prev_sample: torch.Tensor  # 先前时间步的样本
    pred_original_sample: Optional[torch.Tensor] = None  # 可选的预测去噪样本，默认为None


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 扩散时间步的数量
    max_beta=0.999,  # 使用的最大beta值，防止出现奇点
    alpha_transform_type="cosine",  # alpha_bar的噪声调度类型，默认为"cosine"
):
    """
    创建一个beta调度，离散化给定的alpha_t_bar函数，该函数定义了
    （1-beta）随时间的累积乘积，从 t = [0,1]。

    包含一个alpha_bar函数，该函数接受参数t并将其转换为
    该部分扩散过程的（1-beta）的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 要生成的betas数量。
        max_beta (`float`): 要使用的最大beta值；使用小于1的值以
                     防止奇点。
        alpha_transform_type (`str`, *可选*，默认为`cosine`): alpha_bar的噪声调度类型。
                     从`cosine`或`exp`中选择。

    返回：
        betas (`np.ndarray`): 调度器用于逐步模型输出的betas
    """
    # 检查指定的 alpha 变换类型是否为 "cosine"
        if alpha_transform_type == "cosine":
            # 定义 alpha_bar_fn 函数，使用余弦函数进行变换
            def alpha_bar_fn(t):
                # 计算余弦值的平方，生成平滑过渡的 alpha 值
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    
        # 检查指定的 alpha 变换类型是否为 "exp"
        elif alpha_transform_type == "exp":
            # 定义 alpha_bar_fn 函数，使用指数函数进行变换
            def alpha_bar_fn(t):
                # 计算指数衰减值，返回随着时间衰减的 alpha 值
                return math.exp(t * -12.0)
    
        # 如果 alpha_transform_type 不在预设选项内，则抛出错误
        else:
            raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    
        # 初始化一个空列表，用于存储 beta 值
        betas = []
        # 遍历每个扩散时间步
        for i in range(num_diffusion_timesteps):
            # 计算当前时间步的比例 t1
            t1 = i / num_diffusion_timesteps
            # 计算下一个时间步的比例 t2
            t2 = (i + 1) / num_diffusion_timesteps
            # 计算 beta 值并添加到列表中，确保不超过 max_beta
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        # 将 beta 列表转换为浮点数张量并返回
        return torch.tensor(betas, dtype=torch.float32)
# 从 diffusers.schedulers.scheduling_ddim 导入 rescale_zero_terminal_snr 方法
def rescale_zero_terminal_snr(betas):
    """
    根据文献 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 对 betas 进行重新缩放，使终端 SNR 为零

    参数:
        betas (`torch.Tensor`):
            用于初始化调度器的 betas。

    返回:
        `torch.Tensor`: 重新缩放后的 betas，使终端 SNR 为零
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算 alphas 的平方根

    # 存储旧值
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 克隆第一个值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 克隆最后一个值

    # 将最后一个时间步移位为零
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 减去最后一个值

    # 缩放使第一个时间步恢复到旧值
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 进行缩放

    # 将 alphas_bar_sqrt 转换回 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方根
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 还原累积乘积
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 拼接第一个值
    betas = 1 - alphas  # 计算 betas

    return betas  # 返回重新缩放后的 betas


class EulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler 调度器。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查看超类文档以获取库为所有调度器实现的通用方法，如加载和保存。
    # 参数说明部分
        Args:
            num_train_timesteps (`int`, defaults to 1000):  # 训练模型的扩散步骤数量，默认为1000。
                The number of diffusion steps to train the model.
            beta_start (`float`, defaults to 0.0001):  # 推理的起始 beta 值，默认为0.0001。
                The starting `beta` value of inference.
            beta_end (`float`, defaults to 0.02):  # 最终 beta 值，默认为0.02。
                The final `beta` value.
            beta_schedule (`str`, defaults to `"linear"`):  # beta 调度方式，默认为“线性”，可选“线性”或“缩放线性”。
                The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
                `linear` or `scaled_linear`.
            trained_betas (`np.ndarray`, *optional*):  # 可选，直接传入 beta 数组以绕过 beta_start 和 beta_end。
                Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
            prediction_type (`str`, defaults to `epsilon`, *optional*):  # 可选，调度函数的预测类型，默认为 `epsilon`。
                Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
                `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
                Video](https://imagen.research.google/video/paper.pdf) paper).
            interpolation_type(`str`, defaults to `"linear"`, *optional*):  # 可选，插值类型用于计算调度去噪步骤的中间 sigma，默认为“线性”。
                The interpolation type to compute intermediate sigmas for the scheduler denoising steps. Should be one of
                `"linear"` or `"log_linear"`.
            use_karras_sigmas (`bool`, *optional*, defaults to `False`):  # 可选，是否在采样过程中使用 Karras sigma，默认为 False。
                Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
                the sigmas are determined according to a sequence of noise levels {σi}.
            timestep_spacing (`str`, defaults to `"linspace"`):  # 时间步缩放方式，默认为“线性空间”。
                The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
                Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
            steps_offset (`int`, defaults to 0):  # 推理步骤的偏移量，默认为0，某些模型可能需要。
                An offset added to the inference steps, as required by some model families.
            rescale_betas_zero_snr (`bool`, defaults to `False`):  # 可选，是否将 beta 重新缩放为零终端 SNR，默认为 False。
                Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
                dark samples instead of limiting it to samples with medium brightness. Loosely related to
                [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
            final_sigmas_type (`str`, defaults to `"zero"`):  # 最终 sigma 值在采样过程中的噪声调度，默认为“零”。
                The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
                sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
        """
    
        # 兼容的调度器列表，提取 KarrasDiffusionSchedulers 中的名称
        _compatibles = [e.name for e in KarrasDiffusionSchedulers]
        # 设定顺序，通常用于调度器的优先级
        order = 1
    
        # 注册到配置，允许将该方法与配置系统相连
        @register_to_config
    # 初始化函数，设置参数以构建模型
        def __init__(
            # 训练时间步的数量，默认为1000
            self,
            num_train_timesteps: int = 1000,
            # beta的起始值，默认为0.0001
            beta_start: float = 0.0001,
            # beta的结束值，默认为0.02
            beta_end: float = 0.02,
            # beta的调度方式，默认为线性
            beta_schedule: str = "linear",
            # 训练的beta值，默认为None
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
            # 预测类型，默认为"epsilon"
            prediction_type: str = "epsilon",
            # 插值类型，默认为线性
            interpolation_type: str = "linear",
            # 是否使用Karras的sigma，默认为False
            use_karras_sigmas: Optional[bool] = False,
            # sigma的最小值，默认为None
            sigma_min: Optional[float] = None,
            # sigma的最大值，默认为None
            sigma_max: Optional[float] = None,
            # 时间步间距类型，默认为"linspace"
            timestep_spacing: str = "linspace",
            # 时间步类型，默认为"discrete"，可选"discrete"或"continuous"
            timestep_type: str = "discrete",  # can be "discrete" or "continuous"
            # 步骤偏移量，默认为0
            steps_offset: int = 0,
            # 是否重新缩放beta以实现零SNR，默认为False
            rescale_betas_zero_snr: bool = False,
            # 最终sigma的类型，默认为"zero"，可选"zero"或"sigma_min"
            final_sigmas_type: str = "zero",  # can be "zero" or "sigma_min"
        ):
            # 如果提供了训练的beta值，将其转换为张量
            if trained_betas is not None:
                self.betas = torch.tensor(trained_betas, dtype=torch.float32)
            # 如果beta调度方式为线性，则生成线性beta值
            elif beta_schedule == "linear":
                self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
            # 如果beta调度方式为scaled_linear，则生成特定的beta值
            elif beta_schedule == "scaled_linear":
                # 该调度非常特定于潜在扩散模型
                self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            # 如果beta调度方式为squaredcos_cap_v2，则使用Glide余弦调度生成beta值
            elif beta_schedule == "squaredcos_cap_v2":
                self.betas = betas_for_alpha_bar(num_train_timesteps)
            # 如果不支持的调度方式，抛出错误
            else:
                raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")
    
            # 如果需要重新缩放beta以实现零SNR，则执行缩放
            if rescale_betas_zero_snr:
                self.betas = rescale_zero_terminal_snr(self.betas)
    
            # 计算alpha值，等于1减去beta值
            self.alphas = 1.0 - self.betas
            # 计算alpha的累积乘积
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
            # 如果需要重新缩放beta以实现零SNR，调整最后一个alpha值以避免inf
            if rescale_betas_zero_snr:
                # 近乎0但不为0，以避免第一个sigma为inf
                self.alphas_cumprod[-1] = 2**-24
    
            # 计算sigma值，通过公式得到并反转
            sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).flip(0)
            # 创建时间步数组，从0到num_train_timesteps - 1
            timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
            # 将时间步转换为张量
            timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
    
            # 可设值，初始化为None
            self.num_inference_steps = None
    
            # TODO: 支持所有预测类型和时间步类型的完整EDM缩放
            # 如果时间步类型为连续且预测类型为v_prediction，则计算时间步
            if timestep_type == "continuous" and prediction_type == "v_prediction":
                self.timesteps = torch.Tensor([0.25 * sigma.log() for sigma in sigmas])
            # 否则直接使用预定义的时间步
            else:
                self.timesteps = timesteps
    
            # 将sigma与零张量连接以形成最终的sigma
            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    
            # 标记输入缩放是否被调用
            self.is_scale_input_called = False
            # 设置是否使用Karras sigma的标志
            self.use_karras_sigmas = use_karras_sigmas
    
            # 初始化步骤索引和开始索引为None
            self._step_index = None
            self._begin_index = None
            # 将sigma移至CPU以减少CPU和GPU之间的通信
            self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
    
        # 设为属性
        @property
    # 初始化噪声标准差
    def init_noise_sigma(self):
        # 获取初始噪声分布的标准差
        max_sigma = max(self.sigmas) if isinstance(self.sigmas, list) else self.sigmas.max()
        # 如果时间步长间隔配置为 "linspace" 或 "trailing"，返回最大标准差
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return max_sigma
        # 否则返回 (max_sigma**2 + 1) 的平方根
        return (max_sigma**2 + 1) ** 0.5

    # 当前时间步的索引计数器属性
    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加1。
        """
        return self._step_index

    # 第一个时间步的索引属性
    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过 `set_begin_index` 方法从管道中设置。
        """
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制的函数
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的开始索引。此函数应在推理之前从管道运行。

        参数:
            begin_index (`int`):
                调度器的开始索引。
        """
        # 设置调度器的开始索引
        self._begin_index = begin_index

    # 缩放模型输入以确保与调度器的互换性
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器互换性。通过 `(sigma**2 + 1) ** 0.5` 缩放去噪模型输入以匹配 Euler 算法。

        参数:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。

        返回:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 如果步索引为空，初始化步索引
        if self.step_index is None:
            self._init_step_index(timestep)

        # 获取当前时间步的标准差
        sigma = self.sigmas[self.step_index]
        # 按照公式缩放样本输入
        sample = sample / ((sigma**2 + 1) ** 0.5)

        # 标记输入缩放已被调用
        self.is_scale_input_called = True
        # 返回缩放后的样本
        return sample

    # 设置时间步的函数
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
    # 将 sigma 转换为时间步
    def _sigma_to_t(self, sigma, log_sigmas):
        # 获取对数 sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # 计算分布
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 获取 sigma 范围的低索引
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        # 高索引为低索引加1
        high_idx = low_idx + 1

        # 获取低和高的对数 sigma
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 插值计算 sigma
        w = (low - log_sigma) / (low - high)
        # 限制插值权重在0和1之间
        w = np.clip(w, 0, 1)

        # 将插值转换为时间范围
        t = (1 - w) * low_idx + w * high_idx
        # 重新塑形为与 sigma 形状相同
        t = t.reshape(sigma.shape)
        # 返回转换后的时间步
        return t
    # 从指定的 GitHub 链接复制的代码
        def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
            """构造 Karras 等人 (2022) 的噪声调度。"""
    
            # 确保其他调度器复制此函数时不会出现问题的 hack
            # TODO: 将此逻辑添加到其他调度器中
            if hasattr(self.config, "sigma_min"):
                # 从配置中获取 sigma_min 值
                sigma_min = self.config.sigma_min
            else:
                # 如果没有，则设置为 None
                sigma_min = None
    
            if hasattr(self.config, "sigma_max"):
                # 从配置中获取 sigma_max 值
                sigma_max = self.config.sigma_max
            else:
                # 如果没有，则设置为 None
                sigma_max = None
    
            # 如果 sigma_min 为 None，则取 in_sigmas 的最后一个值
            sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
            # 如果 sigma_max 为 None，则取 in_sigmas 的第一个值
            sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()
    
            rho = 7.0  # 在论文中使用的值
            # 创建一个线性空间，从 0 到 1，共 num_inference_steps 个点
            ramp = np.linspace(0, 1, num_inference_steps)
            # 计算 sigma_min 和 sigma_max 的逆 rho 次方
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            # 计算新的 sigma 值
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            # 返回计算得到的 sigma 值
            return sigmas
    
        def index_for_timestep(self, timestep, schedule_timesteps=None):
            # 如果没有提供 schedule_timesteps，则使用类中的 timesteps
            if schedule_timesteps is None:
                schedule_timesteps = self.timesteps
    
            # 查找与 timestep 匹配的索引
            indices = (schedule_timesteps == timestep).nonzero()
    
            # 第一个 `step` 的 sigma 索引总是第二个索引
            # 如果只有一个，则使用最后一个
            # 这样可以确保在去噪调度中间开始时不跳过 sigma
            pos = 1 if len(indices) > 1 else 0
    
            # 返回所需的索引值
            return indices[pos].item()
    
        def _init_step_index(self, timestep):
            # 如果 begin_index 为 None
            if self.begin_index is None:
                # 如果 timestep 是张量，则将其移动到 timesteps 的设备
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.to(self.timesteps.device)
                # 根据 timestep 初始化步骤索引
                self._step_index = self.index_for_timestep(timestep)
            else:
                # 否则，使用 _begin_index
                self._step_index = self._begin_index
    
        def step(
            self,
            model_output: torch.Tensor,
            timestep: Union[float, torch.Tensor],
            sample: torch.Tensor,
            s_churn: float = 0.0,
            s_tmin: float = 0.0,
            s_tmax: float = float("inf"),
            s_noise: float = 1.0,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
        def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 与 original_samples 具有相同的设备和数据类型
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备类型是否为 "mps" 且 timesteps 是否为浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64，因此将 timesteps 转换为 float32
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将 schedule_timesteps 转换为与 original_samples 相同的设备类型
            schedule_timesteps = self.timesteps.to(original_samples.device)
            # 将 timesteps 转换为与 original_samples 相同的设备类型
            timesteps = timesteps.to(original_samples.device)

        # 当 scheduler 用于训练时，self.begin_index 为 None，或者管道未实现 set_begin_index
        if self.begin_index is None:
            # 根据 timesteps 和 schedule_timesteps 计算步长索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一次去噪步骤后调用 add_noise（用于修补）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一次去噪步骤之前调用 add_noise，以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 根据步长索引从 sigmas 中提取相应的 sigma 值并展平
        sigma = sigmas[step_indices].flatten()
        # 如果 sigma 的形状小于 original_samples 的形状，则增加维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 计算带噪声的样本，原始样本加上噪声乘以 sigma
        noisy_samples = original_samples + noise * sigma
        # 返回带噪声的样本
        return noisy_samples
    # 计算给定样本、噪声和时间步长的速度
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # 检查 timesteps 是否为整数类型
        if (
            isinstance(timesteps, int)
            or isinstance(timesteps, torch.IntTensor)
            or isinstance(timesteps, torch.LongTensor)
        ):
            # 抛出错误，提示不支持整数作为时间步
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.get_velocity()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )
    
        # 如果使用 MPS 设备且时间步是浮点数类型
        if sample.device.type == "mps" and torch.is_floating_point(timesteps):
            # MPS 不支持 float64 类型，转换为 float32
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timesteps = timesteps.to(sample.device, dtype=torch.float32)
        else:
            # 将时间步转换为当前设备的默认类型
            schedule_timesteps = self.timesteps.to(sample.device)
            timesteps = timesteps.to(sample.device)
    
        # 获取每个时间步对应的索引
        step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        # 将 alpha 累积乘积转换到样本的设备
        alphas_cumprod = self.alphas_cumprod.to(sample)
        # 计算 sqrt(alpha) 的积
        sqrt_alpha_prod = alphas_cumprod[step_indices] ** 0.5
        # 将结果展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 扩展维度，直到与样本形状匹配
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
        # 计算 sqrt(1 - alpha) 的积
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[step_indices]) ** 0.5
        # 将结果展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 扩展维度，直到与样本形状匹配
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
        # 计算速度，结合噪声和样本
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回计算得到的速度
        return velocity
    
    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```