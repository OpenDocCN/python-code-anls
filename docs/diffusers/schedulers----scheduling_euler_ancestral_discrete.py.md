# `.\diffusers\schedulers\scheduling_euler_ancestral_discrete.py`

```py
# 版权所有 2024 Katherine Crowson 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（"许可证"）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按 "原样" 提供的，
# 不附带任何形式的担保或条件，无论是明示或暗示的。
# 有关许可证所规定的权限和限制的具体信息，请参阅许可证。

import math  # 导入数学库以进行数学运算
from dataclasses import dataclass  # 从数据类模块导入 dataclass 装饰器
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的类

import numpy as np  # 导入 NumPy 库以进行数组和矩阵运算
import torch  # 导入 PyTorch 库以进行张量运算

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具中导入混合类和注册函数
from ..utils import BaseOutput, logging  # 从工具中导入基本输出类和日志记录功能
from ..utils.torch_utils import randn_tensor  # 从工具中导入生成随机张量的函数
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin  # 从调度工具中导入调度器类

logger = logging.get_logger(__name__)  # 初始化日志记录器，使用当前模块的名称

@dataclass
# 从 diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput 复制而来，DDPM->EulerAncestralDiscrete
class EulerAncestralDiscreteSchedulerOutput(BaseOutput):
    """
    调度器的 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像)：
            计算出的前一时间步的样本 `(x_{t-1})`。`prev_sample` 应作为下一模型输入使用
            在去噪循环中。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像)：
            基于当前时间步的模型输出预测的去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或指导。
    """

    prev_sample: torch.Tensor  # 前一时间步的样本
    pred_original_sample: Optional[torch.Tensor] = None  # 可选的预测去噪样本

# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制而来
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 扩散时间步的数量
    max_beta=0.999,  # 使用的最大 beta 值
    alpha_transform_type="cosine",  # alpha_bar 的噪声调度类型
):
    """
    创建一个 beta 调度，离散化给定的 alpha_t_bar 函数，该函数定义了
    在 t = [0,1] 之间随时间变化的 (1-beta) 的累积乘积。

    包含一个 alpha_bar 函数，该函数接受参数 t，并将其转换为到达
    扩散过程的该部分的 (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以
                     防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择。

    返回：
        betas (`np.ndarray`): 调度器用于推进模型输出的 beta 值
    """
    # 检查指定的 alpha 变换类型是否为 "cosine"
        if alpha_transform_type == "cosine":
    
            # 定义一个函数 alpha_bar_fn，用于计算 cos 变换
            def alpha_bar_fn(t):
                # 根据 t 计算 cos 变换值的平方
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    
        # 检查指定的 alpha 变换类型是否为 "exp"
        elif alpha_transform_type == "exp":
    
            # 定义一个函数 alpha_bar_fn，用于计算指数变换
            def alpha_bar_fn(t):
                # 根据 t 计算 e 的负 12 倍乘以 t 的指数值
                return math.exp(t * -12.0)
    
        # 如果 alpha 变换类型不受支持，抛出异常
        else:
            raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    
        # 初始化一个空列表，用于存储 beta 值
        betas = []
        # 遍历每个扩散时间步
        for i in range(num_diffusion_timesteps):
            # 计算当前时间步的归一化值
            t1 = i / num_diffusion_timesteps
            # 计算下一个时间步的归一化值
            t2 = (i + 1) / num_diffusion_timesteps
            # 计算 beta 值并添加到列表中，确保不超过 max_beta
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        # 返回 beta 值的张量，数据类型为 float32
        return torch.tensor(betas, dtype=torch.float32)
# 从 diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr 复制而来
def rescale_zero_terminal_snr(betas):
    """
    根据 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 重新缩放 betas，使其终端信噪比为零

    参数:
        betas (`torch.Tensor`):
            初始化调度器时使用的 betas。

    返回:
        `torch.Tensor`: 重新缩放后的 betas，终端信噪比为零
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算累积乘积的平方根

    # 存储旧值。
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 复制第一个值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 复制最后一个值

    # 将最后一个时间步的值移位为零。
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 减去最后一个值

    # 缩放，使第一个时间步返回到旧值。
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 按比例调整

    # 将 alphas_bar_sqrt 转换为 betas
    alphas_bar = alphas_bar_sqrt**2  # 恢复平方
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 恢复累积乘积
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个值添加到结果
    betas = 1 - alphas  # 计算 betas

    return betas  # 返回重新缩放后的 betas


class EulerAncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    使用欧拉方法步骤进行祖先采样。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关所有调度器通用方法的文档，请检查超类文档，例如加载和保存。
    # 参数说明部分，用于描述类初始化时的参数及其默认值
    Args:
        # 训练模型的扩散步骤数量，默认值为1000
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        # 推理的起始beta值，默认值为0.0001
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        # 推理的最终beta值，默认值为0.02
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        # beta的调度方式，指定beta范围到一系列beta的映射，默认值为"linear"
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        # 可选参数，直接传入betas数组以绕过beta_start和beta_end
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        # 预测类型，默认值为`epsilon`，可选参数
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        # 时间步的缩放方式，默认值为"linspace"
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        # 在推理步骤中添加的偏移量，默认值为0
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        # 是否将betas重新缩放以达到零终端信噪比，默认值为False
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    # 兼容的调度器列表，从KarrasDiffusionSchedulers中提取名称
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置调度器的顺序，默认为1
    order = 1

    # 用于注册配置的初始化方法
    @register_to_config
    def __init__(
        # 训练步骤数量，默认值为1000
        num_train_timesteps: int = 1000,
        # 起始beta值，默认值为0.0001
        beta_start: float = 0.0001,
        # 结束beta值，默认值为0.02
        beta_end: float = 0.02,
        # beta调度类型，默认值为"linear"
        beta_schedule: str = "linear",
        # 可选的训练beta值，默认为None
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 预测类型，默认为`epsilon`
        prediction_type: str = "epsilon",
        # 时间步缩放类型，默认为"linspace"
        timestep_spacing: str = "linspace",
        # 步骤偏移量，默认为0
        steps_offset: int = 0,
        # 是否重新缩放beta以获得零终端SNR，默认为False
        rescale_betas_zero_snr: bool = False,
    ):
        # 检查是否提供了训练过的贝塔值
        if trained_betas is not None:
            # 将训练过的贝塔值转换为 PyTorch 张量，数据类型为 float32
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果贝塔调度为线性
        elif beta_schedule == "linear":
            # 生成从 beta_start 到 beta_end 的线性间隔贝塔值，数量为 num_train_timesteps
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果贝塔调度为缩放线性
        elif beta_schedule == "scaled_linear":
            # 该调度特别针对潜在扩散模型
            # 生成从 beta_start 的平方根到 beta_end 的平方根的线性间隔，然后平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 如果贝塔调度为平方余弦调度版本 2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果未实现的贝塔调度
        else:
            # 抛出未实现的错误
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 如果需要将贝塔值重新缩放到零 SNR
        if rescale_betas_zero_snr:
            # 对贝塔值进行重新缩放
            self.betas = rescale_zero_terminal_snr(self.betas)

        # 计算 alpha 值，等于 1 减去贝塔值
        self.alphas = 1.0 - self.betas
        # 计算累积乘积的 alpha 值
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 如果需要将贝塔值重新缩放到零 SNR
        if rescale_betas_zero_snr:
            # 将最后一个累积乘积 alpha 值设置为接近 0 的小值，以避免无穷大
            # FP16 最小正数次正规值在这里表现良好
            self.alphas_cumprod[-1] = 2**-24

        # 计算 sigma 值，基于 alpha 值的公式
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        # 反转 sigma 数组，并追加一个 0.0，转换为 float32 类型
        sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
        # 将 sigma 转换为 PyTorch 张量
        self.sigmas = torch.from_numpy(sigmas)

        # 可设置的属性
        self.num_inference_steps = None
        # 生成从 0 到 num_train_timesteps - 1 的时间步长数组，并反转
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=float)[::-1].copy()
        # 将时间步长转换为 PyTorch 张量
        self.timesteps = torch.from_numpy(timesteps)
        # 初始化标记，指示是否调用了输入缩放
        self.is_scale_input_called = False

        # 初始化步骤索引
        self._step_index = None
        # 初始化起始索引
        self._begin_index = None
        # 将 sigma 移动到 CPU，以避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def init_noise_sigma(self):
        # 返回初始噪声分布的标准差
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            # 返回 sigma 值的最大值
            return self.sigmas.max()

        # 返回最大 sigma 的平方加 1 的平方根
        return (self.sigmas.max() ** 2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度步骤后增加 1。
        """
        # 返回步骤索引
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应该通过 `set_begin_index` 方法从管道中设置。
        """
        # 返回起始索引
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的起始索引。此函数应在推理之前从管道中运行。

        参数:
            begin_index (`int`):
                调度器的起始索引。
        """
        # 设置调度器的起始索引
        self._begin_index = begin_index
    # 定义一个方法来缩放模型输入，接受样本和时间步
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        # 方法说明，确保与调度器兼容，根据当前时间步缩放去噪模型输入
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.
    
        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.
    
        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
    
        # 如果当前步索引为 None，初始化步索引
        if self.step_index is None:
            self._init_step_index(timestep)
    
        # 获取当前步的 sigma 值
        sigma = self.sigmas[self.step_index]
        # 根据公式缩放输入样本
        sample = sample / ((sigma**2 + 1) ** 0.5)
        # 标记输入缩放函数已被调用
        self.is_scale_input_called = True
        # 返回缩放后的样本
        return sample
    # 设置离散时间步长，用于扩散链（在推理之前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步长（在推理之前运行）。
    
        参数:
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数。
            device (`str` 或 `torch.device`, *可选*):
                要移动时间步长的设备。如果为 `None`，则不移动时间步长。
        """
        # 存储推理步骤的数量
        self.num_inference_steps = num_inference_steps
    
        # 根据配置的时间步长间隔计算时间步长
        if self.config.timestep_spacing == "linspace":
            # 创建从0到num_train_timesteps-1的均匀间隔的时间步长，并反向排序
            timesteps = np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[
                ::-1
            ].copy()
        elif self.config.timestep_spacing == "leading":
            # 计算步骤比率，创建整数时间步长
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # 通过比率创建整数时间步长，避免当num_inference_step为3的幂时出现问题
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            # 添加步骤偏移量
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            # 计算步骤比率，创建整数时间步长
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # 通过比率创建整数时间步长，避免当num_inference_step为3的幂时出现问题
            timesteps = (np.arange(self.config.num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
            # 减去1以调整时间步长
            timesteps -= 1
        else:
            # 抛出错误，指示不支持的时间步长间隔类型
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )
    
        # 计算sigmas，基于累积alpha值
        sigmas = np.array(((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5)
        # 使用时间步长插值sigmas
        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)
        # 将sigmas与0.0连接并转换为float32类型
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        # 将sigmas转换为torch张量，并移动到指定设备
        self.sigmas = torch.from_numpy(sigmas).to(device=device)
    
        # 将时间步长转换为torch张量，并移动到指定设备
        self.timesteps = torch.from_numpy(timesteps).to(device=device)
        # 初始化步骤索引和开始索引为None
        self._step_index = None
        self._begin_index = None
        # 将sigmas移动到CPU，以避免过多的CPU/GPU通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
    
        # 从diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep复制的代码
    # 定义一个方法，根据给定的时间步索引找到对应的索引
        def index_for_timestep(self, timestep, schedule_timesteps=None):
            # 如果没有提供调度时间步，则使用默认时间步
            if schedule_timesteps is None:
                schedule_timesteps = self.timesteps
    
            # 找到与当前时间步匹配的所有索引
            indices = (schedule_timesteps == timestep).nonzero()
    
            # 对于第一步，选择第二个索引（或仅有一个时选择第一个），以避免跳过
            pos = 1 if len(indices) > 1 else 0
    
            # 返回找到的索引值
            return indices[pos].item()
    
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制的代码
        def _init_step_index(self, timestep):
            # 如果开始索引为 None，初始化步骤索引
            if self.begin_index is None:
                # 如果时间步是张量，则将其转移到相同的设备上
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.to(self.timesteps.device)
                # 根据时间步查找步骤索引
                self._step_index = self.index_for_timestep(timestep)
            else:
                # 如果已定义开始索引，则使用它
                self._step_index = self._begin_index
    
        # 定义一个步骤方法，接受模型输出、时间步、样本等参数
        def step(
            self,
            model_output: torch.Tensor,
            timestep: Union[float, torch.Tensor],
            sample: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise 复制的代码
        def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 与 original_samples 的设备和数据类型相同
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备是否为 "mps" 且 timesteps 是否为浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64，因此将时间步长转换为 float32
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 否则，直接将时间步长转换为与 original_samples 相同的设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # 当 scheduler 用于训练时，self.begin_index 为 None，或 pipeline 未实现 set_begin_index
        if self.begin_index is None:
            # 根据时间步长计算对应的步骤索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一个去噪步骤后调用 add_noise（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一个去噪步骤之前调用 add_noise 以创建初始潜在图像 (img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 根据步骤索引提取 sigmas，并展平为一维
        sigma = sigmas[step_indices].flatten()
        # 确保 sigma 的形状与 original_samples 的形状相匹配
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 生成带噪声的样本，通过 original_samples 和噪声加权 sigma 进行叠加
        noisy_samples = original_samples + noise * sigma
        # 返回带噪声的样本
        return noisy_samples

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```