# `.\diffusers\schedulers\scheduling_dpm_cogvideox.py`

```py
# 版权所有 2024 CogVideoX 团队，清华大学、ZhipuAI 和 HuggingFace 团队。
# 保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件
# 以“原样”基础提供，不附带任何明示或暗示的担保或条件。
# 有关许可证所管辖的权限和限制的具体说明，请参见许可证。

# 免责声明：此代码受到 https://github.com/pesser/pytorch_diffusion 和
# https://github.com/hojonathanho/diffusion 的强烈影响

# 导入数学模块
import math
# 从数据类库导入 dataclass 装饰器
from dataclasses import dataclass
# 导入类型相关的类
from typing import List, Optional, Tuple, Union

# 导入 numpy 库
import numpy as np
# 导入 torch 库
import torch

# 从配置工具导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具库导入 BaseOutput 类
from ..utils import BaseOutput
# 从 torch 工具库导入随机张量生成函数
from ..utils.torch_utils import randn_tensor
# 从调度工具库导入 KarrasDiffusionSchedulers 和 SchedulerMixin
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

@dataclass
# 从 diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput 中复制，DDPM 改为 DDIM
class DDIMSchedulerOutput(BaseOutput):
    """
    调度器的 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像):
            先前时间步的计算样本 `(x_{t-1})`。 `prev_sample` 应在去噪循环中用作下一个模型输入。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)` 的图像):
            基于当前时间步模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或指导。
    """

    # 先前时间步的样本张量
    prev_sample: torch.Tensor
    # 可选的预测原始样本张量
    pred_original_sample: Optional[torch.Tensor] = None

# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建一个 beta 调度，离散化给定的 alpha_t_bar 函数，该函数定义了
    随时间推移 (1-beta) 的累积乘积，从 t = [0,1]。

    包含一个函数 alpha_bar，该函数接受一个参数 t，并将其转换为
    在该部分扩散过程中 (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta；使用小于 1 的值
                     防止奇异性。
        alpha_transform_type (`str`，*可选*，默认为 `cosine`): alpha_bar 的噪声调度类型。
                     选择 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于推动模型输出的 betas
    # 文档字符串，描述该代码块的功能
    """
    # 检查 alpha_transform_type 是否为 "cosine"
    if alpha_transform_type == "cosine":

        # 定义 alpha_bar_fn 函数，计算基于余弦的变换
        def alpha_bar_fn(t):
            # 计算余弦函数的平方，调整时间参数 t
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 检查 alpha_transform_type 是否为 "exp"
    elif alpha_transform_type == "exp":

        # 定义 alpha_bar_fn 函数，计算基于指数的变换
        def alpha_bar_fn(t):
            # 计算指数衰减函数
            return math.exp(t * -12.0)

    # 如果 alpha_transform_type 不符合预期，抛出错误
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化一个空列表，用于存储 beta 值
    betas = []
    # 遍历所有的扩散时间步数
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步的归一化值 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步的归一化值 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值并添加到列表中，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 beta 列表转换为 PyTorch 张量，数据类型为 float32
    return torch.tensor(betas, dtype=torch.float32)
# 定义一个函数，用于重新缩放 beta，使其终端信噪比为零
def rescale_zero_terminal_snr(alphas_cumprod):
    """
    根据 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 重新缩放 beta 使终端信噪比为零

    参数:
        betas (`torch.Tensor`):
            初始化调度器时使用的 betas。

    返回:
        `torch.Tensor`: 具有零终端信噪比的重新缩放的 betas
    """

    # 计算累积 alpha 的平方根
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # 存储旧值
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 记录第一个 alpha 的平方根
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 记录最后一个 alpha 的平方根

    # 将最后一个时间步的值移至零
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # 缩放以将第一个时间步的值恢复到旧值
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # 将 alphas_bar_sqrt 转换为 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方根

    # 返回重新缩放后的 alpha 值
    return alphas_bar


# 定义一个类，继承自 SchedulerMixin 和 ConfigMixin，扩展去噪过程
class CogVideoXDPMScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDIMScheduler` 扩展了在去噪扩散概率模型 (DDPM) 中引入的去噪过程，增加了非马尔可夫指导。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。查看超类文档以获取库为所有调度器实现的通用方法，例如加载和保存。
    """

    # 兼容的调度器名称列表
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置调度器的顺序
    order = 1

    @register_to_config
    # 初始化方法，定义调度器的参数
    def __init__(
        self,
        num_train_timesteps: int = 1000,  # 训练时间步的数量
        beta_start: float = 0.00085,  # beta 的起始值
        beta_end: float = 0.0120,  # beta 的结束值
        beta_schedule: str = "scaled_linear",  # beta 调度的类型
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 训练好的 betas（可选）
        clip_sample: bool = True,  # 是否裁剪样本
        set_alpha_to_one: bool = True,  # 是否将 alpha 设置为 1
        steps_offset: int = 0,  # 时间步的偏移量
        prediction_type: str = "epsilon",  # 预测类型
        clip_sample_range: float = 1.0,  # 裁剪样本的范围
        sample_max_value: float = 1.0,  # 样本的最大值
        timestep_spacing: str = "leading",  # 时间步间隔的类型
        rescale_betas_zero_snr: bool = False,  # 是否重新缩放 betas 使终端 SNR 为零
        snr_shift_scale: float = 3.0,  # SNR 移位比例
    ):
        # 检查是否有训练好的 beta 值
        if trained_betas is not None:
            # 将训练好的 beta 值转换为张量，数据类型为 float32
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果 beta_schedule 是 "linear"
        elif beta_schedule == "linear":
            # 在给定范围内生成线性间隔的 beta 值
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果 beta_schedule 是 "scaled_linear"
        elif beta_schedule == "scaled_linear":
            # 该调度特定于潜在扩散模型
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float64) ** 2
        # 如果 beta_schedule 是 "squaredcos_cap_v2"
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果以上情况都不符合
        else:
            # 抛出未实现的错误
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alpha 值
        self.alphas = 1.0 - self.betas
        # 计算累积 alpha 值
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 修改：根据 SD3 调整 SNR
        self.alphas_cumprod = self.alphas_cumprod / (snr_shift_scale + (1 - snr_shift_scale) * self.alphas_cumprod)

        # 如果需要为零 SNR 重新缩放
        if rescale_betas_zero_snr:
            # 重新缩放累积 alpha 值
            self.alphas_cumprod = rescale_zero_terminal_snr(self.alphas_cumprod)

        # 在每个 ddim 步骤中，查看前一个累积 alpha 值
        # 对于最后一步，没有前一个累积 alpha 值，因为已经在 0 了
        # `set_alpha_to_one` 决定是否将该参数设置为 1，或使用“非前一个”的最终 alpha
        self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # 初始噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的值
        self.num_inference_steps = None
        # 创建时间步长的反向序列
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    # 获取特定时间步的方差
    def _get_variance(self, timestep, prev_timestep):
        # 当前时间步的累积 alpha 值
        alpha_prod_t = self.alphas_cumprod[timestep]
        # 前一个时间步的累积 alpha 值（如果有效）
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        # 当前时间步的 beta 值
        beta_prod_t = 1 - alpha_prod_t
        # 前一个时间步的 beta 值
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 根据 beta 和 alpha 计算方差
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # 返回计算出的方差
        return variance

    # 对模型输入进行缩放，确保与调度器的兼容性
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器互换性。

        Args:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *optional*):
                扩散链中的当前时间步。

        Returns:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 返回输入样本（未进行实际缩放）
        return sample
    # 定义设置推理步数的方法，接受推理步骤数和设备参数
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步（在推理之前运行）。
    
        参数：
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数，基于预训练模型。
        """
    
        # 检查推理步骤数是否超过训练时间步数
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} 不能大于 `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps}，因为与此调度器一起训练的 unet 模型只能处理"
                f" 最大 {self.config.num_train_timesteps} 时间步。"
            )
    
        # 将推理步骤数赋值给实例变量
        self.num_inference_steps = num_inference_steps
    
        # 根据配置的时间步间隔类型生成时间步
        if self.config.timestep_spacing == "linspace":
            # 生成线性间隔的时间步并反向排序
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            # 计算步长比例
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # 通过乘以比例生成整数时间步，避免出现问题
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            # 添加步骤偏移量
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            # 计算步长比例
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # 通过乘以比例生成整数时间步，避免出现问题
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            # 减去1以调整时间步
            timesteps -= 1
        else:
            # 抛出错误，如果时间步间隔不被支持
            raise ValueError(
                f"{self.config.timestep_spacing} 不被支持。请确保选择 'leading' 或 'trailing'。"
            )
    
        # 将生成的时间步转换为张量并移动到指定设备
        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    # 定义获取变量的方法，计算相关的 lambda 值
    def get_variables(self, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back=None):
        # 计算当前时间步的 lambda 值
        lamb = ((alpha_prod_t / (1 - alpha_prod_t)) ** 0.5).log()
        # 计算前一个时间步的 lambda 值
        lamb_next = ((alpha_prod_t_prev / (1 - alpha_prod_t_prev)) ** 0.5).log()
        # 计算两个 lambda 之间的差
        h = lamb_next - lamb
    
        # 如果提供了反向时间步的 alpha 值，进行进一步计算
        if alpha_prod_t_back is not None:
            lamb_previous = ((alpha_prod_t_back / (1 - alpha_prod_t_back)) ** 0.5).log()
            # 计算最后一个 h 值
            h_last = lamb - lamb_previous
            # 计算比率 r
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            # 返回 h 和 lambda 值
            return h, None, lamb, lamb_next
    # 计算多重输出的辅助函数
    def get_mult(self, h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back):
        # 计算第一个乘数，使用当前和前一个 alpha 值
        mult1 = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * (-h).exp()
        # 计算第二个乘数，基于 h 和前一个 alpha 值
        mult2 = (-2 * h).expm1() * alpha_prod_t_prev**0.5

        # 如果 alpha_prod_t_back 不是 None，则计算额外的乘数
        if alpha_prod_t_back is not None:
            # 计算第三个乘数，基于 r
            mult3 = 1 + 1 / (2 * r)
            # 计算第四个乘数，基于 r
            mult4 = 1 / (2 * r)
            # 返回四个乘数
            return mult1, mult2, mult3, mult4
        else:
            # 返回前两个乘数
            return mult1, mult2

    # 步进函数，处理模型输出和样本
    def step(
        self,
        model_output: torch.Tensor,  # 模型输出的张量
        old_pred_original_sample: torch.Tensor,  # 之前的原始样本预测
        timestep: int,  # 当前时间步
        timestep_back: int,  # 回溯时间步
        sample: torch.Tensor,  # 输入样本的张量
        eta: float = 0.0,  # 附加参数，默认为 0
        use_clipped_model_output: bool = False,  # 是否使用剪辑后的模型输出
        generator=None,  # 随机数生成器
        variance_noise: Optional[torch.Tensor] = None,  # 可选的方差噪声
        return_dict: bool = False,  # 是否以字典形式返回
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的函数
    def add_noise(
        self,
        original_samples: torch.Tensor,  # 原始样本的张量
        noise: torch.Tensor,  # 噪声的张量
        timesteps: torch.IntTensor,  # 时间步的张量
    ) -> torch.Tensor:  # 返回添加噪声后的张量
        # 确保 alphas_cumprod 和 timestep 与 original_samples 具有相同的设备和数据类型
        # 将 self.alphas_cumprod 移动到设备，以避免后续 add_noise 调用时重复的 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为与 original_samples 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到 original_samples 的设备
        timesteps = timesteps.to(original_samples.device)

        # 计算当前时间步的 alpha 乘积的平方根
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 展平平方根的 alpha 乘积
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于原始样本的维度，则添加额外的维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算当前时间步的 1 - alpha 乘积的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 展平平方根的 1 - alpha 乘积
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于原始样本的维度，则添加额外的维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算带噪声的样本
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回带噪声的样本
        return noisy_samples

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity 复制的函数
    # 定义一个获取速度的函数，接受样本、噪声和时间步作为输入，返回速度张量
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timesteps 与样本的设备和数据类型相同
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        # 将 alphas_cumprod 的数据类型转换为样本的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        # 将 timesteps 转移到样本的设备
        timesteps = timesteps.to(sample.device)
    
        # 计算 sqrt(alpha_prod) 的平方根，基于给定的时间步
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将其展平为一维张量
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果平方根的形状小于样本的形状，则在最后一个维度添加维度
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
        # 计算 sqrt(1 - alpha_prod) 的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将其展平为一维张量
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果平方根的形状小于样本的形状，则在最后一个维度添加维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
        # 计算速度，结合噪声和样本的影响
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回计算出的速度
        return velocity
    
    # 定义一个返回训练时间步数的函数
    def __len__(self):
        # 返回配置中的训练时间步数
        return self.config.num_train_timesteps
```