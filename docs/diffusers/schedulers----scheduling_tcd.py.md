# `.\diffusers\schedulers\scheduling_tcd.py`

```py
# 版权声明，说明此代码的所有权归斯坦福大学团队和HuggingFace团队所有
# 
# 根据Apache许可证，版本2.0（“许可证”）进行许可；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件
# 根据许可证分发是在“按原样”基础上进行的，
# 不提供任何形式的担保或条件，包括明示或暗示的。
# 请参见许可证以获取有关特定语言的权限和
# 限制的详细信息。

# 免责声明：此代码受到以下项目的强烈影响 https://github.com/pesser/pytorch_diffusion
# 和 https://github.com/hojonathanho/diffusion

import math  # 导入数学模块以进行数学运算
from dataclasses import dataclass  # 从数据类模块导入dataclass装饰器
from typing import List, Optional, Tuple, Union  # 从typing模块导入类型提示

import numpy as np  # 导入numpy库以进行数值计算
import torch  # 导入PyTorch库以进行张量操作

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入配置混合类和注册函数
from ..schedulers.scheduling_utils import SchedulerMixin  # 从调度工具导入调度混合类
from ..utils import BaseOutput, logging  # 从工具模块导入基本输出类和日志记录工具
from ..utils.torch_utils import randn_tensor  # 从PyTorch工具导入随机张量生成函数

logger = logging.get_logger(__name__)  # 创建一个日志记录器以记录当前模块的日志，禁用pylint的命名警告

@dataclass  # 将类标记为数据类
class TCDSchedulerOutput(BaseOutput):  # 定义调度器输出类，继承基本输出类
    """
    调度器的`step`函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为`(batch_size, num_channels, height, width)`的图像):
            先前时间步的计算样本`(x_{t-1})`。`prev_sample`应作为去噪循环中的下一个模型输入使用。
        pred_noised_sample (`torch.Tensor`，形状为`(batch_size, num_channels, height, width)`的图像):
            基于当前时间步的模型输出的预测噪声样本`(x_{s})`。
    """

    prev_sample: torch.Tensor  # 定义先前样本属性，类型为torch张量
    pred_noised_sample: Optional[torch.Tensor] = None  # 定义可选的预测噪声样本属性，默认值为None

# 从diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar复制
def betas_for_alpha_bar(  # 定义用于生成beta调度的函数
    num_diffusion_timesteps,  # 传播时间步数的参数
    max_beta=0.999,  # 最大beta的默认值
    alpha_transform_type="cosine",  # alpha变换类型的默认值为“余弦”
):
    """
    创建一个beta调度，离散化给定的alpha_t_bar函数，该函数定义了时间上(1-beta)的累积乘积，从t = [0,1]。

    包含一个alpha_bar函数，该函数接受一个参数t并将其转换为(1-beta)的累积乘积
    到该传播过程的部分。

    参数：
        num_diffusion_timesteps (`int`): 要生成的beta数量。
        max_beta (`float`): 要使用的最大beta；使用小于1的值以防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认值为`cosine`): alpha_bar的噪声调度类型。
                     从`cosine`或`exp`中选择

    返回：
        betas (`np.ndarray`): 调度程序用于模型输出的步进的beta。
    """
    # 检查 alpha_transform_type 是否为 "cosine"
    if alpha_transform_type == "cosine":

        # 定义 alpha_bar_fn 函数，计算余弦平方值
        def alpha_bar_fn(t):
            # 根据公式计算余弦函数值的平方，调整时间参数 t
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 检查 alpha_transform_type 是否为 "exp"
    elif alpha_transform_type == "exp":

        # 定义 alpha_bar_fn 函数，计算指数衰减值
        def alpha_bar_fn(t):
            # 计算指数函数的衰减值，使用负的时间参数乘以常数
            return math.exp(t * -12.0)

    # 如果 alpha_transform_type 不是已支持的类型，抛出错误
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化空列表，用于存储 beta 值
    betas = []
    # 循环遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步的归一化值 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步的归一化值 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值，并将其添加到列表中，取最小值以限制
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 beta 列表转换为张量并返回，数据类型为浮点32
    return torch.tensor(betas, dtype=torch.float32)
# 从 diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr 复制而来
def rescale_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """
    根据 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 重新缩放 betas，使其具有零终端 SNR

    参数:
        betas (`torch.Tensor`):
            调度器初始化时使用的 betas。

    返回:
        `torch.Tensor`: 具有零终端 SNR 的重新缩放 betas
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas，取 1 减去 betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算 alphas 的平方根

    # 存储旧值。
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 克隆第一个 alphas_bar_sqrt 的值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 克隆最后一个 alphas_bar_sqrt 的值

    # 移动，使最后一个时间步为零。
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 从每个值中减去最后一个值

    # 缩放，使第一个时间步恢复到旧值。
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 根据旧值进行缩放

    # 将 alphas_bar_sqrt 转换为 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方根
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 还原累积乘积
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 连接第一个值与后续值
    betas = 1 - alphas  # 计算 betas

    return betas  # 返回重新缩放的 betas


class TCDScheduler(SchedulerMixin, ConfigMixin):
    """
    `TCDScheduler` 结合了论文 `Trajectory Consistency Distillation` 中提出的 `Strategic Stochastic Sampling`，
    扩展了原始的多步一致性采样，允许无限制的轨迹遍历。

    该代码基于 TCD 的官方仓库 (https://github.com/jabir-zheng/TCD)。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。 [`~ConfigMixin`] 负责存储在调度器的 `__init__` 函数中传递的所有配置属性，
    如 `num_train_timesteps`。可以通过 `scheduler.config.num_train_timesteps` 访问。[`SchedulerMixin`] 提供通用的加载和保存
    功能，通过 [`SchedulerMixin.save_pretrained`] 和 [`~SchedulerMixin.from_pretrained`] 函数实现。

    """

    order = 1  # 设置调度器的顺序为 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,  # 训练时间步的数量，默认为 1000
        beta_start: float = 0.00085,  # beta 的起始值，默认为 0.00085
        beta_end: float = 0.012,  # beta 的结束值，默认为 0.012
        beta_schedule: str = "scaled_linear",  # beta 的调度策略，默认为 "scaled_linear"
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 已训练的 betas，默认为 None
        original_inference_steps: int = 50,  # 原始推断步骤的数量，默认为 50
        clip_sample: bool = False,  # 是否裁剪样本，默认为 False
        clip_sample_range: float = 1.0,  # 样本裁剪范围，默认为 1.0
        set_alpha_to_one: bool = True,  # 是否将 alpha 设置为 1，默认为 True
        steps_offset: int = 0,  # 步骤偏移量，默认为 0
        prediction_type: str = "epsilon",  # 预测类型，默认为 "epsilon"
        thresholding: bool = False,  # 是否使用阈值处理，默认为 False
        dynamic_thresholding_ratio: float = 0.995,  # 动态阈值比例，默认为 0.995
        sample_max_value: float = 1.0,  # 样本的最大值，默认为 1.0
        timestep_spacing: str = "leading",  # 时间步间隔类型，默认为 "leading"
        timestep_scaling: float = 10.0,  # 时间步缩放因子，默认为 10.0
        rescale_betas_zero_snr: bool = False,  # 是否重新缩放 betas 以实现零 SNR，默认为 False
    ):
        # 检查已训练的贝塔值是否为 None
        if trained_betas is not None:
            # 将训练好的贝塔值转换为张量，数据类型为 float32
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果贝塔调度方式为线性
        elif beta_schedule == "linear":
            # 创建从 beta_start 到 beta_end 的线性贝塔值
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果贝塔调度方式为缩放线性
        elif beta_schedule == "scaled_linear":
            # 该调度方式特定于潜在扩散模型
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 如果贝塔调度方式为 squaredcos_cap_v2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果以上都不符合，抛出未实现错误
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 针对零 SNR 进行重新缩放
        if rescale_betas_zero_snr:
            # 调整贝塔值以适应零终端 SNR
            self.betas = rescale_zero_terminal_snr(self.betas)

        # 计算 alphas，作为 1 减去贝塔值
        self.alphas = 1.0 - self.betas
        # 计算 alphas 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 在每一步的 ddim 中查看之前的 alphas_cumprod
        # 最后一步时没有前一个 alphas_cumprod，因为我们已经在 0 处
        # `set_alpha_to_one` 决定是否将此参数设置为 1 或使用最后一个非前驱的 alpha
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的值
        self.num_inference_steps = None
        # 反转的时间步长数组
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
        self.custom_timesteps = False

        # 初始化步长和起始索引
        self._step_index = None
        self._begin_index = None

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep 复制的函数
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果未提供调度时间步，则使用默认时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与当前时间步匹配的索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 第一步的 sigma 索引总是第二个索引（如果只有一个则为最后一个）
        # 确保在去噪调度中不意外跳过 sigma
        pos = 1 if len(indices) > 1 else 0

        # 返回所需索引的值
        return indices[pos].item()

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制的函数
    def _init_step_index(self, timestep):
        # 如果 begin_index 为 None
        if self.begin_index is None:
            # 如果时间步是张量，转移到相应设备
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 初始化步长索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则，使用预设的起始索引
            self._step_index = self._begin_index

    @property
    # 定义一个方法，返回当前的步骤索引
        def step_index(self):
            return self._step_index
    
        # 定义一个属性，返回开始索引
        @property
        def begin_index(self):
            """
            返回第一个时间步的索引。应通过 `set_begin_index` 方法从管道设置。
            """
            return self._begin_index
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制而来
        # 定义一个方法，用于设置调度器的开始索引
        def set_begin_index(self, begin_index: int = 0):
            """
            设置调度器的开始索引。此函数应在推理之前从管道运行。
    
            参数:
                begin_index (`int`):
                    调度器的开始索引。
            """
            self._begin_index = begin_index
    
        # 定义一个方法，确保与需要根据当前时间步缩放去噪模型输入的调度器互操作
        def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
            """
            确保与需要根据当前时间步缩放去噪模型输入的调度器互操作。
    
            参数:
                sample (`torch.Tensor`):
                    输入样本。
                timestep (`int`, *可选*):
                    扩散链中的当前时间步。
    
            返回:
                `torch.Tensor`:
                    一个缩放后的输入样本。
            """
            return sample
    
        # 从 diffusers.schedulers.scheduling_ddim.DDIMScheduler._get_variance 复制而来
        # 定义一个方法，计算当前和前一个时间步的方差
        def _get_variance(self, timestep, prev_timestep):
            # 获取当前时间步的累积 alpha 值
            alpha_prod_t = self.alphas_cumprod[timestep]
            # 获取前一个时间步的累积 alpha 值，如果无效则使用最终累积值
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
            # 计算当前时间步的 beta 值
            beta_prod_t = 1 - alpha_prod_t
            # 计算前一个时间步的 beta 值
            beta_prod_t_prev = 1 - alpha_prod_t_prev
    
            # 根据公式计算方差
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    
            return variance
    
        # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制而来
    # 定义动态阈值采样的方法，输入为样本张量，返回处理后的张量
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "动态阈值处理：在每个采样步骤中，我们将 s 设置为 xt0（在时间步 t 的 x_0 预测）中的某个百分位绝对像素值，
        如果 s > 1，则将 xt0 阈值处理为范围 [-s, s]，然后除以 s。动态阈值处理将饱和像素（接近 -1 和 1 的像素）
        向内推，这样可以在每一步主动防止像素饱和。我们发现动态阈值处理显著提高了照片真实感以及图像-文本对齐，
        尤其是在使用非常大的引导权重时。"

        https://arxiv.org/abs/2205.11487
        """
        # 获取输入样本的数值类型
        dtype = sample.dtype
        # 获取样本的批次大小、通道数和剩余维度
        batch_size, channels, *remaining_dims = sample.shape

        # 如果数据类型不是 float32 或 float64，则转换为 float 类型
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 为了进行分位数计算而上升数据类型，并且 clamp 对 CPU half 类型未实现

        # 将样本展平，以便对每个图像进行分位数计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        # 计算样本的绝对值
        abs_sample = sample.abs()  # "某个百分位绝对像素值"

        # 计算绝对样本的指定百分位
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 将 s 限制在 [1, sample_max_value] 的范围内
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当限制到 min=1 时，相当于标准剪切到 [-1, 1]
        # 在维度 1 上扩展 s 的维度，以便广播
        s = s.unsqueeze(1)  # (batch_size, 1) 因为 clamp 会在维度 0 上广播
        # 将样本限制在 [-s, s] 范围内，然后除以 s
        sample = torch.clamp(sample, -s, s) / s  # "我们将 xt0 阈值处理为 [-s, s] 的范围，然后除以 s"

        # 重新调整样本的形状为原来的形状
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原来的数据类型
        sample = sample.to(dtype)

        # 返回处理后的样本
        return sample

    # 定义设置时间步的方法，接受多个参数用于配置
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,  # 推理步骤的数量
        device: Union[str, torch.device] = None,  # 设备类型（字符串或 torch 设备）
        original_inference_steps: Optional[int] = None,  # 原始推理步骤数量
        timesteps: Optional[List[int]] = None,  # 时间步列表
        strength: float = 1.0,  # 强度参数，默认为 1.0
    # 定义步骤的方法，处理模型输出与样本
    def step(
        self,
        model_output: torch.Tensor,  # 模型输出的张量
        timestep: int,  # 当前时间步
        sample: torch.Tensor,  # 输入样本张量
        eta: float = 0.3,  # 噪声参数，默认为 0.3
        generator: Optional[torch.Generator] = None,  # 可选的随机数生成器
        return_dict: bool = True,  # 是否返回字典格式的结果，默认为 True
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的内容
    def add_noise(
        self,
        original_samples: torch.Tensor,  # 原始样本的张量
        noise: torch.Tensor,  # 噪声的张量
        timesteps: torch.IntTensor,  # 时间步的整数张量
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 的设备和数据类型与 original_samples 相同
        # 将 self.alphas_cumprod 移动到设备上，以避免后续 add_noise 调用时的冗余 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为与 original_samples 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到 original_samples 的设备上
        timesteps = timesteps.to(original_samples.device)

        # 计算 sqrt_alpha_prod 为 alphas_cumprod 在 timesteps 位置的平方根
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 扁平化
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 original_samples，则增加维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 sqrt_one_minus_alpha_prod 为 (1 - alphas_cumprod 在 timesteps 位置) 的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 扁平化
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 original_samples，则增加维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算带噪声的样本 noisy_samples
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回带噪声的样本
        return noisy_samples

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity 复制而来
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 的设备和数据类型与 sample 相同
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        # 将 alphas_cumprod 转换为与 sample 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        # 将 timesteps 移动到 sample 的设备上
        timesteps = timesteps.to(sample.device)

        # 计算 sqrt_alpha_prod 为 alphas_cumprod 在 timesteps 位置的平方根
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 扁平化
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 sample，则增加维度
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 sqrt_one_minus_alpha_prod 为 (1 - alphas_cumprod 在 timesteps 位置) 的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 扁平化
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 sample，则增加维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算速度 velocity
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回速度
        return velocity

    # 返回训练时间步数的数量
    def __len__(self):
        return self.config.num_train_timesteps

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.previous_timestep 复制而来
    # 定义一个方法，用于获取给定时间步的前一个时间步
    def previous_timestep(self, timestep):
        # 检查是否存在自定义时间步
        if self.custom_timesteps:
            # 找到当前时间步在时间步数组中的索引
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            # 如果当前时间步是最后一个，设置前一个时间步为 -1
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            # 否则，获取下一个时间步
            else:
                prev_t = self.timesteps[index + 1]
        # 如果没有自定义时间步
        else:
            # 计算推理步骤的数量，优先使用实例属性，否则使用配置的训练时间步
            num_inference_steps = (
                self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
            )
            # 计算前一个时间步
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps
    
        # 返回前一个时间步
        return prev_t
```