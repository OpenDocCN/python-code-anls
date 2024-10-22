# `.\diffusers\schedulers\scheduling_lcm.py`

```py
# 版权所有 2024 斯坦福大学团队与 HuggingFace 团队，保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件按“原样”提供，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证所涵盖的权限和限制的具体信息，请参见许可证。

# 免责声明：此代码受 https://github.com/pesser/pytorch_diffusion
# 和 https://github.com/hojonathanho/diffusion 的强烈影响

import math  # 导入数学库以执行数学运算
from dataclasses import dataclass  # 从数据类模块导入数据类装饰器
from typing import List, Optional, Tuple, Union  # 从 typing 模块导入类型提示

import numpy as np  # 导入 NumPy 库以进行数值计算
import torch  # 导入 PyTorch 库以使用张量和深度学习功能

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入配置混合和注册函数
from ..utils import BaseOutput, logging  # 从工具模块导入基础输出类和日志功能
from ..utils.torch_utils import randn_tensor  # 从工具模块导入随机张量生成函数
from .scheduling_utils import SchedulerMixin  # 从调度工具导入调度混合器

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

@dataclass
class LCMSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像)：
            先前时间步的计算样本 `(x_{t-1})`。`prev_sample` 应作为下一个模型输入用于
            去噪循环。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像)：
            基于当前时间步的模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或指导。
    """

    prev_sample: torch.Tensor  # 存储先前样本的张量
    denoised: Optional[torch.Tensor] = None  # 可选的去噪样本，默认为 None

# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 参数：扩散时间步的数量
    max_beta=0.999,  # 参数：使用的最大 beta 值，避免奇异性应低于 1
    alpha_transform_type="cosine",  # 参数：alpha_bar 的噪声调度类型，默认为“余弦”
):
    """
    创建一个 beta 调度，离散化给定的 alpha_t_bar 函数，该函数定义了
    (1-beta) 随时间的累积乘积，范围从 t = [0,1]。

    包含一个 alpha_bar 函数，该函数接受参数 t，并将其转换为
    扩散过程的 (1-beta) 累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 要生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     可选择 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于步骤模型输出的 beta 值
    """
    # 检查给定的 alpha 转换类型是否为“cosine”
        if alpha_transform_type == "cosine":
    
            # 定义用于计算 alpha_bar 的函数，基于余弦函数
            def alpha_bar_fn(t):
                # 计算并返回余弦函数值的平方
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    
        # 检查 alpha 转换类型是否为“exp”
        elif alpha_transform_type == "exp":
    
            # 定义用于计算 alpha_bar 的函数，基于指数函数
            def alpha_bar_fn(t):
                # 计算并返回指数衰减值
                return math.exp(t * -12.0)
    
        # 如果 alpha 转换类型不支持，则抛出异常
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
# 从 diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr 中复制
def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """
    将 betas 重新缩放为零终端 SNR，基于 https://arxiv.org/pdf/2305.08891.pdf (算法 1)

    参数:
        betas (`torch.Tensor`):
            初始化调度器时使用的 betas。

    返回:
        `torch.Tensor`: 具有零终端 SNR 的重新缩放的 betas
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas，即 1 减去 betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算累积乘积的平方根

    # 存储旧值。
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 记录第一个 alphas_bar_sqrt 的值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 记录最后一个 alphas_bar_sqrt 的值

    # 将最后一个时间步的值移为零。
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 从所有值中减去最后一个值，使其为零

    # 缩放以将第一个时间步恢复到旧值。
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 进行缩放以恢复初始值

    # 将 alphas_bar_sqrt 转换回 betas
    alphas_bar = alphas_bar_sqrt**2  # 将平方根恢复为平方
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 通过累积乘积的逆运算恢复 alphas
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个 alphas_bar 的值添加到结果中
    betas = 1 - alphas  # 计算 betas，即 1 减去 alphas

    return betas  # 返回重新缩放的 betas


class LCMScheduler(SchedulerMixin, ConfigMixin):
    """
    `LCMScheduler` 扩展了在去噪扩散概率模型 (DDPM) 中引入的去噪程序，并实现了
    非马尔可夫引导。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。[`~ConfigMixin`] 负责存储在调度器的
    `__init__` 函数中传入的所有配置属性，例如 `num_train_timesteps`。它们可以通过
    `scheduler.config.num_train_timesteps` 访问。[`SchedulerMixin`] 提供通用的加载和保存
    功能，通过 [`SchedulerMixin.save_pretrained`] 和 [`~SchedulerMixin.from_pretrained`] 函数。
    """

    order = 1  # 定义调度器的顺序

    @register_to_config  # 注册到配置中，允许该函数的参数在配置中存储
    def __init__(
        self,
        num_train_timesteps: int = 1000,  # 训练时间步的数量，默认为 1000
        beta_start: float = 0.00085,  # beta 的起始值，默认为 0.00085
        beta_end: float = 0.012,  # beta 的结束值，默认为 0.012
        beta_schedule: str = "scaled_linear",  # beta 的调度方式，默认为 "scaled_linear"
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 经过训练的 betas，默认为 None
        original_inference_steps: int = 50,  # 原始推理步骤数，默认为 50
        clip_sample: bool = False,  # 是否剪裁样本，默认为 False
        clip_sample_range: float = 1.0,  # 剪裁样本的范围，默认为 1.0
        set_alpha_to_one: bool = True,  # 是否将 alpha 设置为 1，默认为 True
        steps_offset: int = 0,  # 步骤偏移量，默认为 0
        prediction_type: str = "epsilon",  # 预测类型，默认为 "epsilon"
        thresholding: bool = False,  # 是否应用阈值处理，默认为 False
        dynamic_thresholding_ratio: float = 0.995,  # 动态阈值处理的比例，默认为 0.995
        sample_max_value: float = 1.0,  # 样本最大值，默认为 1.0
        timestep_spacing: str = "leading",  # 时间步间距，默认为 "leading"
        timestep_scaling: float = 10.0,  # 时间步缩放因子，默认为 10.0
        rescale_betas_zero_snr: bool = False,  # 是否重新缩放 betas 以实现零 SNR，默认为 False
    ):
        # 检查已训练的 beta 值是否存在
        if trained_betas is not None:
            # 将训练好的 beta 值转换为张量，数据类型为浮点32
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 检查 beta 调度是否为线性
        elif beta_schedule == "linear":
            # 生成从 beta_start 到 beta_end 的线性间隔的 beta 值
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 检查 beta 调度是否为 scaled_linear
        elif beta_schedule == "scaled_linear":
            # 该调度非常特定于潜在扩散模型
            # 生成 beta_start 和 beta_end 的平方根线性间隔的 beta 值，然后平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 检查 beta 调度是否为 squaredcos_cap_v2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            # 使用 betas_for_alpha_bar 函数生成 beta 值
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            # 抛出未实现错误，如果 beta_schedule 未被实现
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 如果需要，重新缩放为零 SNR
        if rescale_betas_zero_snr:
            # 调用 rescale_zero_terminal_snr 函数重新缩放 beta 值
            self.betas = rescale_zero_terminal_snr(self.betas)

        # 计算 alpha 值，alpha = 1 - beta
        self.alphas = 1.0 - self.betas
        # 计算 alpha 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 在每个 ddim 步骤中，查看之前的 alphas_cumprod
        # 最后一步时，因为已经在 0，所以没有前一个 alphas_cumprod
        # set_alpha_to_one 决定是否将该参数简单地设置为 1，或使用“非前一个”的最终 alpha
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置值
        self.num_inference_steps = None
        # 生成反向的时间步长张量
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        # 标记是否自定义时间步长
        self.custom_timesteps = False

        # 初始化步骤索引和开始索引
        self._step_index = None
        self._begin_index = None

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep 复制而来
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果未提供调度时间步长，则使用默认时间步长
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与给定时间步长相等的索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 对于第一个“步骤”，选择的 sigma 索引始终是第二个索引
        # （如果只有一个，则为最后一个索引），确保不会意外跳过 sigma
        pos = 1 if len(indices) > 1 else 0

        # 返回指定位置的索引值
        return indices[pos].item()

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制而来
    def _init_step_index(self, timestep):
        # 如果 begin_index 为空，初始化步骤索引
        if self.begin_index is None:
            # 如果时间步长是张量，将其转换为相同设备
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 使用 index_for_timestep 方法设置步骤索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则将步骤索引设置为开始索引
            self._step_index = self._begin_index

    # 定义一个属性
    # 定义一个方法，返回当前的步骤索引
    def step_index(self):
        # 返回私有属性 _step_index
        return self._step_index

    # 定义一个只读属性，表示起始时间步的索引
    @property
    def begin_index(self):
        """
        起始时间步的索引。应通过管道使用 `set_begin_index` 方法设置。
        """
        # 返回私有属性 _begin_index
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制的
    # 定义一个方法，设置调度器的起始索引
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的起始索引。此函数应在推理前通过管道运行。

        参数:
            begin_index (`int`):
                调度器的起始索引。
        """
        # 将传入的起始索引值赋给私有属性 _begin_index
        self._begin_index = begin_index

    # 定义一个方法，确保与需要根据当前时间步缩放去噪模型输入的调度器的互换性
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器的互换性。

        参数:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。
        返回:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 直接返回输入样本，未进行任何缩放处理
        return sample

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制的
    # 定义私有方法 _threshold_sample，输入为一个张量 sample，返回处理后的张量
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "动态阈值处理：在每个采样步骤中，我们将 s 设置为 xt0（在时间步 t 时 x_0 的预测）的某个百分位绝对像素值，
        如果 s > 1，则将 xt0 阈值化到范围 [-s, s]，然后除以 s。动态阈值处理将饱和像素（接近 -1 和 1 的像素）向内推，
        从而主动防止每个步骤的像素饱和。我们发现动态阈值处理显著改善了照片真实感以及图像-文本对齐，特别是在使用非常大的
        引导权重时。"

        https://arxiv.org/abs/2205.11487
        """
        # 获取样本的数据类型
        dtype = sample.dtype
        # 获取样本的批量大小、通道数及其余维度
        batch_size, channels, *remaining_dims = sample.shape

        # 如果样本的类型不是 float32 或 float64，则将其转换为 float
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 为了进行分位数计算而上升精度，且 CPU 半精度不支持 clamping

        # 将样本展平，以便沿每个图像进行分位数计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        # 计算样本的绝对值
        abs_sample = sample.abs()  # "某个百分位绝对像素值"

        # 计算每个样本的动态阈值 s
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 将 s 限制在指定范围内
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当下限为 1 时，相当于标准裁剪到 [-1, 1]
        # 将 s 的维度扩展为 (batch_size, 1)，以便在维度 0 上广播
        s = s.unsqueeze(1)  # (batch_size, 1)
        # 对样本进行裁剪并标准化
        sample = torch.clamp(sample, -s, s) / s  # "将 xt0 阈值化到范围 [-s, s]，然后除以 s"

        # 将样本的形状还原为原来的维度
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本的类型转换回原来的数据类型
        sample = sample.to(dtype)

        # 返回处理后的样本
        return sample

    # 定义设置时间步的函数
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,  # 可选参数：推理步骤数
        device: Union[str, torch.device] = None,  # 可选参数：设备类型
        original_inference_steps: Optional[int] = None,  # 可选参数：原始推理步骤数
        timesteps: Optional[List[int]] = None,  # 可选参数：时间步列表
        strength: int = 1.0,  # 强度参数，默认为 1.0
    # 定义用于边界条件离散的缩放函数
    def get_scalings_for_boundary_condition_discrete(self, timestep):
        # 设置默认的 sigma 数据值
        self.sigma_data = 0.5  # 默认值：0.5
        # 根据时间步和配置进行缩放
        scaled_timestep = timestep * self.config.timestep_scaling

        # 计算跳过的系数
        c_skip = self.sigma_data**2 / (scaled_timestep**2 + self.sigma_data**2)
        # 计算输出的系数
        c_out = scaled_timestep / (scaled_timestep**2 + self.sigma_data**2) ** 0.5
        # 返回跳过的系数和输出的系数
        return c_skip, c_out

    # 定义步进方法
    def step(
        self,
        model_output: torch.Tensor,  # 模型输出的张量
        timestep: int,  # 当前的时间步
        sample: torch.Tensor,  # 当前的样本张量
        generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
        return_dict: bool = True,  # 是否返回字典格式的结果
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的方法
    def add_noise(
        self,
        original_samples: torch.Tensor,  # 原始样本的张量
        noise: torch.Tensor,  # 噪声张量
        timesteps: torch.IntTensor,  # 时间步的张量
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 的设备和数据类型与 original_samples 一致
        # 将 self.alphas_cumprod 移动到目标设备，以避免后续 add_noise 调用时重复的 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为与 original_samples 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到与 original_samples 相同的设备
        timesteps = timesteps.to(original_samples.device)

        # 计算 sqrt_alpha_prod 为 alphas_cumprod 在 timesteps 索引处的平方根
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平为一维数组
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度少于 original_samples 的维度，添加新的维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 sqrt_one_minus_alpha_prod 为 (1 - alphas_cumprod 在 timesteps 索引处) 的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平为一维数组
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度少于 original_samples 的维度，添加新的维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 根据 sqrt_alpha_prod、original_samples 和 sqrt_one_minus_alpha_prod 计算 noisy_samples
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回噪声样本
        return noisy_samples

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity 复制的代码
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 的设备和数据类型与 sample 一致
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        # 将 alphas_cumprod 转换为与 sample 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        # 将 timesteps 移动到与 sample 相同的设备
        timesteps = timesteps.to(sample.device)

        # 计算 sqrt_alpha_prod 为 alphas_cumprod 在 timesteps 索引处的平方根
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平为一维数组
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度少于 sample 的维度，添加新的维度
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 sqrt_one_minus_alpha_prod 为 (1 - alphas_cumprod 在 timesteps 索引处) 的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平为一维数组
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度少于 sample 的维度，添加新的维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 根据 sqrt_alpha_prod 和 noise 计算 velocity，并减去 sqrt_one_minus_alpha_prod 和 sample 的乘积
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回速度
        return velocity

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.previous_timestep 复制的代码
    # 定义一个方法，用于获取给定时间步的前一个时间步
        def previous_timestep(self, timestep):
            # 如果有自定义的时间步
            if self.custom_timesteps:
                # 找到当前时间步在时间步数组中的索引位置
                index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
                # 如果当前索引是最后一个时间步的索引
                if index == self.timesteps.shape[0] - 1:
                    # 设置前一个时间步为 -1，表示没有前一个时间步
                    prev_t = torch.tensor(-1)
                else:
                    # 否则，取当前索引后一个时间步的值作为前一个时间步
                    prev_t = self.timesteps[index + 1]
            else:
                # 如果没有自定义时间步，计算推理步骤的数量
                num_inference_steps = (
                    self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
                )
                # 计算前一个时间步，基于当前时间步和推理步骤
                prev_t = timestep - self.config.num_train_timesteps // num_inference_steps
    
            # 返回计算得到的前一个时间步
            return prev_t
```