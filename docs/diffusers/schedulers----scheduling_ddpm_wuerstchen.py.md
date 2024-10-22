# `.\diffusers\schedulers\scheduling_ddpm_wuerstchen.py`

```py
# 版权声明，标明版权所有者和许可证信息
# Copyright (c) 2022 Pablo Pernías MIT License
# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 (“许可证”) 进行许可；
# 除非遵循许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，分发的软件是按“原样”提供的，
# 不提供任何形式的保证或条件。
# 请参阅许可证以了解特定语言所管辖的权限和限制。

# 免责声明：此文件受到 https://github.com/ermongroup/ddim 的强烈影响

# 导入数学模块
import math
# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入类型相关的类和接口
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch

# 从配置和注册相关的模块导入类
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入 BaseOutput 类
from ..utils import BaseOutput
# 从 PyTorch 工具模块导入 randn_tensor 函数
from ..utils.torch_utils import randn_tensor
# 从调度工具模块导入 SchedulerMixin 类
from .scheduling_utils import SchedulerMixin


# 定义调度器输出类，包含上一时间步的样本
@dataclass
class DDPMWuerstchenSchedulerOutput(BaseOutput):
    """
    调度器步骤函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            计算得到的前一时间步样本 (x_{t-1})。`prev_sample` 应在去噪循环中作为下一个模型输入使用。
    """

    # 前一时间步的样本
    prev_sample: torch.Tensor


# 定义生成 beta 值的函数，基于 alpha_t_bar 函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建一个 beta 调度，该调度离散化给定的 alpha_t_bar 函数，定义了
    随时间变化的 (1-beta) 的累积乘积，范围为 t = [0,1]。

    包含一个 alpha_bar 函数，该函数接受参数 t 并将其转换为 (1-beta) 的累积乘积
    直到扩散过程的该部分。

    参数：
        num_diffusion_timesteps (`int`): 要生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值
                     以防止奇异性。
        alpha_transform_type (`str`, *可选*，默认为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择。

    返回：
        betas (`np.ndarray`): 调度器用于逐步模型输出的 beta 值
    """
    # 根据 alpha_transform_type 选择不同的 alpha_bar 函数
    if alpha_transform_type == "cosine":

        # 定义基于余弦的 alpha_bar 函数
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        # 定义基于指数的 alpha_bar 函数
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        # 抛出不支持的 alpha_transform_type 错误
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化 beta 列表
    betas = []
    # 生成 beta 值，遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 当前时间步 t1
        t1 = i / num_diffusion_timesteps
        # 下一个时间步 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算并添加 beta 值，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 betas 列表转换为 PyTorch 张量，数据类型为 float32
        return torch.tensor(betas, dtype=torch.float32)
# 定义 DDPMWuerstchenScheduler 类，继承自 SchedulerMixin 和 ConfigMixin
class DDPMWuerstchenScheduler(SchedulerMixin, ConfigMixin):
    """
    Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
    Langevin dynamics sampling.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2006.11239

    Args:
        scaler (`float`): ....
        s (`float`): ....
    """

    # 注册配置函数
    @register_to_config
    def __init__(
        # 初始化方法的参数，scaler 和 s，具有默认值
        self,
        scaler: float = 1.0,
        s: float = 0.008,
    ):
        # 将传入的 scaler 值赋给实例变量
        self.scaler = scaler
        # 将 s 转换为张量并赋值给实例变量
        self.s = torch.tensor([s])
        # 计算初始 alpha 累积乘积，使用 cos 函数公式
        self._init_alpha_cumprod = torch.cos(self.s / (1 + self.s) * torch.pi * 0.5) ** 2

        # 设置初始噪声分布的标准差
        self.init_noise_sigma = 1.0

    # 定义计算 alpha 累积乘积的方法
    def _alpha_cumprod(self, t, device):
        # 根据 scaler 的值调整时间 t
        if self.scaler > 1:
            t = 1 - (1 - t) ** self.scaler
        elif self.scaler < 1:
            t = t**self.scaler
        # 计算 alpha 累积乘积
        alpha_cumprod = torch.cos(
            (t + self.s.to(device)) / (1 + self.s.to(device)) * torch.pi * 0.5
        ) ** 2 / self._init_alpha_cumprod.to(device)
        # 限制 alpha 的范围在 [0.0001, 0.9999]
        return alpha_cumprod.clamp(0.0001, 0.9999)

    # 定义缩放模型输入的方法
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.Tensor`: scaled input sample
        """
        # 直接返回输入样本，暂时未实现缩放逻辑
        return sample

    # 定义设置时间步的方法
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        timesteps: Optional[List[int]] = None,
        device: Union[str, torch.device] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`Dict[float, int]`):
                the number of diffusion steps used when generating samples with a pre-trained model. If passed, then
                `timesteps` must be `None`.
            device (`str` or `torch.device`, optional):
                the device to which the timesteps are moved to. {2 / 3: 20, 0.0: 10}
        """
        # 如果没有提供时间步，则生成从 1.0 到 0.0 的等间距张量
        if timesteps is None:
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device)
        # 如果时间步不是张量，则将其转换为张量并移动到指定设备
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.Tensor(timesteps).to(device)
        # 将时间步赋值给实例变量
        self.timesteps = timesteps
    # 定义一个函数，进行反向扩散过程的预测
    def step(
        self,  # self 引用实例对象
        model_output: torch.Tensor,  # 从学习到的扩散模型直接输出的张量
        timestep: int,  # 当前扩散链中的离散时间步
        sample: torch.Tensor,  # 当前扩散过程中生成的样本实例
        generator=None,  # 随机数生成器，默认为 None
        return_dict: bool = True,  # 是否返回字典，默认为 True
    ) -> Union[DDPMWuerstchenSchedulerOutput, Tuple]:  # 函数返回类型为 DDPMWuerstchenSchedulerOutput 或元组
        """
        通过反向 SDE 预测前一个时间步的样本。核心函数用于从学习到的模型输出传播扩散过程
        （通常是预测的噪声）。

        参数:
            model_output (`torch.Tensor`): 学习到的扩散模型的直接输出。
            timestep (`int`): 当前扩散链中的离散时间步。
            sample (`torch.Tensor`):
                当前扩散过程中生成的样本实例。
            generator: 随机数生成器。
            return_dict (`bool`): 选择返回元组而不是 DDPMWuerstchenSchedulerOutput 类

        返回:
            [`DDPMWuerstchenSchedulerOutput`] 或 `tuple`: 如果 `return_dict` 为 True，返回 [`DDPMWuerstchenSchedulerOutput`]，
            否则返回元组。当返回元组时，第一个元素是样本张量。

        """
        # 获取模型输出的张量的数据类型
        dtype = model_output.dtype
        # 获取模型输出的张量所在设备
        device = model_output.device
        # 将当前时间步赋值给 t
        t = timestep

        # 获取前一个时间步
        prev_t = self.previous_timestep(t)

        # 计算当前时间步的累积 alpha 值，并调整维度
        alpha_cumprod = self._alpha_cumprod(t, device).view(t.size(0), *[1 for _ in sample.shape[1:]])
        # 计算前一个时间步的累积 alpha 值，并调整维度
        alpha_cumprod_prev = self._alpha_cumprod(prev_t, device).view(prev_t.size(0), *[1 for _ in sample.shape[1:]])
        # 计算 alpha 值
        alpha = alpha_cumprod / alpha_cumprod_prev

        # 根据当前样本和模型输出计算均值 mu
        mu = (1.0 / alpha).sqrt() * (sample - (1 - alpha) * model_output / (1 - alpha_cumprod).sqrt())

        # 生成随机噪声张量 std_noise
        std_noise = randn_tensor(mu.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
        # 计算标准差 std
        std = ((1 - alpha) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)).sqrt() * std_noise
        # 预测前一个时间步的样本
        pred = mu + std * (prev_t != 0).float().view(prev_t.size(0), *[1 for _ in sample.shape[1:]])

        # 如果不返回字典，则返回元组形式
        if not return_dict:
            return (pred.to(dtype),)

        # 返回 DDPMWuerstchenSchedulerOutput 对象
        return DDPMWuerstchenSchedulerOutput(prev_sample=pred.to(dtype))

    # 定义一个函数，将噪声添加到原始样本中
    def add_noise(
        self,  # self 引用实例对象
        original_samples: torch.Tensor,  # 原始样本的张量
        noise: torch.Tensor,  # 添加的噪声张量
        timesteps: torch.Tensor,  # 时间步的张量
    ) -> torch.Tensor:  # 函数返回类型为张量
        # 获取原始样本所在的设备
        device = original_samples.device
        # 获取原始样本的数据类型
        dtype = original_samples.dtype
        # 计算给定时间步的累积 alpha 值，并调整维度
        alpha_cumprod = self._alpha_cumprod(timesteps, device=device).view(
            timesteps.size(0), *[1 for _ in original_samples.shape[1:]]
        )
        # 计算添加噪声后的样本
        noisy_samples = alpha_cumprod.sqrt() * original_samples + (1 - alpha_cumprod).sqrt() * noise
        # 返回噪声样本，调整数据类型
        return noisy_samples.to(dtype=dtype)

    # 定义一个函数，返回训练时间步的数量
    def __len__(self):
        # 返回配置中的训练时间步数
        return self.config.num_train_timesteps

    # 定义一个函数，获取前一个时间步
    def previous_timestep(self, timestep):
        # 计算与当前时间步的绝对差值并找到索引
        index = (self.timesteps - timestep[0]).abs().argmin().item()
        # 获取前一个时间步并调整维度
        prev_t = self.timesteps[index + 1][None].expand(timestep.shape[0])
        # 返回前一个时间步
        return prev_t
```