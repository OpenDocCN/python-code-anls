# `.\diffusers\schedulers\scheduling_ddim_parallel.py`

```py
# 版权声明，指定代码作者及版权信息
# Copyright 2024 ParaDiGMS authors and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License 2.0 许可协议，限制文件的使用
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 许可的副本可以在下面的地址获取
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件以 "原样" 基础分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参见许可证以了解特定权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 免责声明：此代码受以下项目强烈影响
# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

# 导入数学库
import math
# 从数据类模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入类型提示相关类型
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch

# 从配置工具导入 ConfigMixin 和注册配置功能
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入 BaseOutput 类
from ..utils import BaseOutput
# 从 PyTorch 工具模块导入生成随机张量的函数
from ..utils.torch_utils import randn_tensor
# 从调度工具导入调度器类
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
# 该类用于表示调度器的步骤函数输出
class DDIMParallelSchedulerOutput(BaseOutput):
    """
    输出类，用于调度器的 `step` 函数输出。

    参数：
        prev_sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)` 的图像):
            上一时间步的计算样本 `(x_{t-1})`。 `prev_sample` 应在去噪循环中用作下一个模型输入。
        pred_original_sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)` 的图像):
            基于当前时间步模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或提供指导。
    """

    # 上一时间步的样本张量
    prev_sample: torch.Tensor
    # 可选的预测去噪样本张量，默认为 None
    pred_original_sample: Optional[torch.Tensor] = None


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 函数复制
def betas_for_alpha_bar(
    # 生成的扩散时间步数量
    num_diffusion_timesteps,
    # 最大 beta 值，防止奇异性
    max_beta=0.999,
    # 噪声调度的类型，默认为 "cosine"
    alpha_transform_type="cosine",
):
    """
    创建一个 beta 调度器，离散化给定的 alpha_t_bar 函数，
    定义了 (1-beta) 随时间的累积乘积，从 t = [0,1] 开始。

    包含一个 alpha_bar 函数，该函数接收 t 参数，并将其转换为
    该扩散过程部分的 (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 要使用的最大 beta；使用小于 1 的值以
                     防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认值为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择。

    返回：
        betas (`np.ndarray`): 调度器用于更新模型输出的 betas
    """
    # 检查 alpha_transform_type 是否为 "cosine"
    if alpha_transform_type == "cosine":

        # 定义一个函数 alpha_bar_fn，接受参数 t
        def alpha_bar_fn(t):
            # 计算余弦函数并返回平方值，用于 alpha 值转换
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 检查 alpha_transform_type 是否为 "exp"
    elif alpha_transform_type == "exp":

        # 定义一个函数 alpha_bar_fn，接受参数 t
        def alpha_bar_fn(t):
            # 计算指数衰减函数，用于 alpha 值转换
            return math.exp(t * -12.0)

    # 如果 alpha_transform_type 不是 "cosine" 或 "exp"，抛出异常
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化一个空列表 betas，用于存储 beta 值
    betas = []
    # 遍历从 0 到 num_diffusion_timesteps 的每个步骤
    for i in range(num_diffusion_timesteps):
        # 计算当前时间 t1，归一化到 [0, 1] 范围
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间 t2，归一化到 [0, 1] 范围
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值，并将其添加到 betas 列表，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 betas 列表转换为 PyTorch 张量，并指定数据类型为 float32
    return torch.tensor(betas, dtype=torch.float32)
# 从 diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr 复制而来
def rescale_zero_terminal_snr(betas):
    """
    根据 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 将 betas 重新缩放为零终端 SNR


    参数：
        betas (`torch.Tensor`):
            初始化调度器所用的 betas。

    返回：
        `torch.Tensor`: 具有零终端 SNR 的重新缩放的 betas
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas，等于 1 减去 betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算 alphas_cumprod 的平方根

    # 存储旧值。
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 克隆第一个 alphas_bar_sqrt 的值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 克隆最后一个 alphas_bar_sqrt 的值

    # 使最后一个时间步的值为零。
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 从 alphas_bar_sqrt 中减去最后一个值

    # 缩放以使第一个时间步返回旧值。
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 根据比例缩放

    # 将 alphas_bar_sqrt 转换回 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方根
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 还原累积乘积
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个 alphas_bar 加入 alphas
    betas = 1 - alphas  # 计算新的 betas，等于 1 减去 alphas

    return betas  # 返回重新缩放的 betas


class DDIMParallelScheduler(SchedulerMixin, ConfigMixin):
    """
    去噪扩散隐式模型是一种调度器，扩展了去噪程序，最初在去噪扩散概率模型 (DDPMs) 中引入了非马尔可夫指导。

    [`~ConfigMixin`] 负责存储传递给调度器 `__init__` 函数的所有配置属性，如 `num_train_timesteps`。
    它们可以通过 `scheduler.config.num_train_timesteps` 访问。
    [`SchedulerMixin`] 通过 [`SchedulerMixin.save_pretrained`] 和
    [`~SchedulerMixin.from_pretrained`] 函数提供一般的加载和保存功能。

    更多详细信息，请参见原始论文：https://arxiv.org/abs/2010.02502

    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]  # 存储兼容的调度器名称
    order = 1  # 设置调度器的顺序
    _is_ode_scheduler = True  # 指示这是一个常微分方程调度器

    @register_to_config
    # 从 diffusers.schedulers.scheduling_ddim.DDIMScheduler.__init__ 复制而来
    def __init__(
        self,
        num_train_timesteps: int = 1000,  # 训练时间步的数量，默认值为 1000
        beta_start: float = 0.0001,  # 起始 beta 值，默认值为 0.0001
        beta_end: float = 0.02,  # 结束 beta 值，默认值为 0.02
        beta_schedule: str = "linear",  # beta 的调度策略，默认值为线性
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 可选的训练 beta 值
        clip_sample: bool = True,  # 是否裁剪样本，默认值为 True
        set_alpha_to_one: bool = True,  # 是否将 alpha 设置为 1，默认值为 True
        steps_offset: int = 0,  # 步骤偏移量，默认值为 0
        prediction_type: str = "epsilon",  # 预测类型，默认值为 epsilon
        thresholding: bool = False,  # 是否使用阈值，默认值为 False
        dynamic_thresholding_ratio: float = 0.995,  # 动态阈值比例，默认值为 0.995
        clip_sample_range: float = 1.0,  # 裁剪样本范围，默认值为 1.0
        sample_max_value: float = 1.0,  # 样本最大值，默认值为 1.0
        timestep_spacing: str = "leading",  # 时间步间隔策略，默认值为 leading
        rescale_betas_zero_snr: bool = False,  # 是否将 beta 重新缩放为零终端 SNR，默认值为 False
    # 定义一个方法，用于设置 beta 值和相关参数
        ):
            # 如果已训练的 beta 值不为 None，则使用这些值
            if trained_betas is not None:
                # 将已训练的 beta 值转换为浮点型张量
                self.betas = torch.tensor(trained_betas, dtype=torch.float32)
            # 如果 beta 调度为线性
            elif beta_schedule == "linear":
                # 生成从 beta_start 到 beta_end 的线性空间 beta 值
                self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
            # 如果 beta 调度为缩放线性
            elif beta_schedule == "scaled_linear":
                # 此调度特定于潜在扩散模型
                self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            # 如果 beta 调度为 squaredcos_cap_v2
            elif beta_schedule == "squaredcos_cap_v2":
                # Glide 余弦调度
                self.betas = betas_for_alpha_bar(num_train_timesteps)
            # 如果 beta 调度不被实现，则抛出异常
            else:
                raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")
    
            # 对于零 SNR 进行重新缩放
            if rescale_betas_zero_snr:
                # 调整 beta 值以满足零终端 SNR 的要求
                self.betas = rescale_zero_terminal_snr(self.betas)
    
            # 计算 alphas，等于 1 减去 beta 值
            self.alphas = 1.0 - self.betas
            # 计算累积乘积的 alphas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
            # 在每一步的 ddim 中，查看先前的 alphas_cumprod
            # 对于最后一步，没有先前的 alphas_cumprod，因为我们已经在 0 处
            # `set_alpha_to_one` 决定是否将此参数设为 1 或使用“非先前” alpha 的最终值
            self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
    
            # 初始化噪声分布的标准差
            self.init_noise_sigma = 1.0
    
            # 可设置的值
            self.num_inference_steps = None
            # 生成反向的时间步长张量
            self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))
    
        # 从 diffusers.schedulers.scheduling_ddim.DDIMScheduler.scale_model_input 复制的方法
        def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
            """
            确保与需要根据当前时间步长缩放去噪模型输入的调度器互换性。
    
            参数：
                sample (`torch.Tensor`):
                    输入样本。
                timestep (`int`, *可选*):
                    扩散链中的当前时间步长。
    
            返回：
                `torch.Tensor`:
                    缩放后的输入样本。
            """
            # 直接返回输入样本，不进行任何缩放
            return sample
    
        def _get_variance(self, timestep, prev_timestep=None):
            # 如果没有提供前一个时间步长，则使用当前时间步长减去的值
            if prev_timestep is None:
                prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    
            # 获取当前时间步长的 alphas 的累积乘积
            alpha_prod_t = self.alphas_cumprod[timestep]
            # 获取前一个时间步长的 alphas 的累积乘积，如果无效则使用最终的 alpha
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
            # 计算当前和前一个时间步长的 beta 值
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
    
            # 计算方差
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    
            # 返回计算得到的方差
            return variance
    # 定义一个批量获取方差的私有方法，接收时间步 t 和前一个时间步 prev_t
        def _batch_get_variance(self, t, prev_t):
            # 获取当前时间步 t 的累积 alpha 值
            alpha_prod_t = self.alphas_cumprod[t]
            # 获取前一个时间步的累积 alpha 值，确保不低于 0
            alpha_prod_t_prev = self.alphas_cumprod[torch.clip(prev_t, min=0)]
            # 对于 prev_t 小于 0 的情况，将其 alpha 值设置为 1.0
            alpha_prod_t_prev[prev_t < 0] = torch.tensor(1.0)
            # 计算当前和前一个时间步的 beta 值
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
    
            # 计算方差，基于 beta 和 alpha 的关系
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    
            # 返回计算出的方差
            return variance
    
        # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制的方法
        def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
            """
            "动态阈值处理：在每个采样步骤中，我们将 s 设置为 xt0（在时间步 t 时对 x_0 的预测）中的某个百分位绝对像素值，
            如果 s > 1，则将 xt0 阈值限制到 [-s, s] 的范围内，然后除以 s。动态阈值处理将饱和像素（接近 -1 和 1 的像素）向内推，
            从而在每个步骤中主动防止像素饱和。我们发现动态阈值处理可以显著提高照片真实感以及更好的图像-文本对齐，
            尤其是在使用非常大的引导权重时。"
            https://arxiv.org/abs/2205.11487
            """
            # 获取输入样本的数据类型
            dtype = sample.dtype
            # 获取输入样本的批量大小、通道数和其他维度
            batch_size, channels, *remaining_dims = sample.shape
    
            # 如果数据类型不是 float32 或 float64，则将样本转换为 float
            if dtype not in (torch.float32, torch.float64):
                sample = sample.float()  # 为分位数计算进行提升，并且没有实现 cpu half 的 clamping
    
            # 将样本展平，以便对每个图像进行分位数计算
            sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
    
            # 计算样本的绝对值
            abs_sample = sample.abs()  # "某个百分位绝对像素值"
    
            # 计算绝对值样本的分位数
            s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
            # 将分位数限制在最小值 1 和最大值 sample_max_value 之间
            s = torch.clamp(
                s, min=1, max=self.config.sample_max_value
            )  # 限制在最小值 1 时，相当于标准裁剪到 [-1, 1]
            # 扩展维度以便后续广播
            s = s.unsqueeze(1)  # (batch_size, 1)，因为 clamp 会在 dim=0 上广播
            # 对样本进行裁剪并除以 s，限制在范围 [-s, s] 内
            sample = torch.clamp(sample, -s, s) / s  # "我们将 xt0 阈值限制到 [-s, s] 的范围内，然后除以 s"
    
            # 将样本形状还原为原始结构
            sample = sample.reshape(batch_size, channels, *remaining_dims)
            # 将样本转换回原始数据类型
            sample = sample.to(dtype)
    
            # 返回处理后的样本
            return sample
    
        # 从 diffusers.schedulers.scheduling_ddim.DDIMScheduler.set_timesteps 复制的方法
    # 定义设置离散时间步长的方法，用于扩散链（在推理之前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步长（在推理之前运行）。

        参数：
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数，基于预训练模型。
        """

        # 检查推理步骤数是否大于训练时的最大时间步数
        if num_inference_steps > self.config.num_train_timesteps:
            # 如果大于，则抛出错误，提示用户不可超过训练时间步数
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        # 将传入的推理步骤数赋值给实例变量
        self.num_inference_steps = num_inference_steps

        # 根据配置的时间步长间隔类型生成时间步数组
        # "linspace", "leading", "trailing" 对应于 https://arxiv.org/abs/2305.08891 表 2 的注释
        if self.config.timestep_spacing == "linspace":
            # 创建一个均匀分布的时间步数组，并进行反转和类型转换
            timesteps = (
                np.linspace(0, self.config.num_train_timesteps - 1, num_inference_steps)
                .round()[::-1]
                .copy()
                .astype(np.int64)
            )
        elif self.config.timestep_spacing == "leading":
            # 计算每个步骤的比例
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # 通过比例生成整数时间步，反转并转换为 int64 类型
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            # 添加偏移量
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            # 计算每个步骤的比例（浮点数）
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # 通过比例生成整数时间步，反转并转换为 int64 类型
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            # 减去 1 以确保时间步正确
            timesteps -= 1
        else:
            # 如果时间步间隔类型不支持，则抛出错误
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'leading' or 'trailing'."
            )

        # 将时间步数组转换为 PyTorch 张量，并移动到指定设备
        self.timesteps = torch.from_numpy(timesteps).to(device)

    # 定义一步推理的方法
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    # 定义批量推理的方法，且不添加噪声
    def batch_step_no_noise(
        self,
        model_output: torch.Tensor,
        timesteps: List[int],
        sample: torch.Tensor,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的内容
    # 定义一个添加噪声的函数，接受原始样本、噪声和时间步长作为输入
    def add_noise(
        self,
        original_samples: torch.Tensor,  # 原始样本的张量
        noise: torch.Tensor,              # 噪声的张量
        timesteps: torch.IntTensor,      # 时间步长的张量
    ) -> torch.Tensor:                   # 返回添加噪声后的张量
        # 确保 alphas_cumprod 和 timesteps 与 original_samples 在相同的设备和数据类型上
        # 将 self.alphas_cumprod 移动到相应设备，以避免后续 add_noise 调用中冗余的 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为与原始样本相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将时间步长移动到与原始样本相同的设备
        timesteps = timesteps.to(original_samples.device)

        # 计算平方根的 alpha 累积乘积
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将其展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 original_samples，则在最后一维扩展
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算平方根的 (1 - alpha) 累积乘积
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将其展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 original_samples，则在最后一维扩展
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算添加噪声后的样本
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回添加噪声后的样本
        return noisy_samples

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity 复制的函数
    # 定义一个获取速度的函数，接受样本、噪声和时间步长作为输入
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timesteps 与 sample 在相同的设备和数据类型上
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        # 将 alphas_cumprod 转换为与样本相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        # 将时间步长移动到与样本相同的设备
        timesteps = timesteps.to(sample.device)

        # 计算平方根的 alpha 累积乘积
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将其展平为一维
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度小于 sample，则在最后一维扩展
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算平方根的 (1 - alpha) 累积乘积
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将其展平为一维
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度小于 sample，则在最后一维扩展
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算速度
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回计算出的速度
        return velocity

    # 定义一个获取训练时间步长数量的函数
    def __len__(self():
        # 返回配置中的训练时间步长数量
        return self.config.num_train_timesteps
```