# `.\diffusers\schedulers\scheduling_repaint.py`

```py
# 版权声明，声明版权所有者及许可协议信息
# Copyright 2024 ETH Zurich Computer Vision Lab and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 许可协议进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 你不得在不遵守许可的情况下使用此文件
# you may not use this file except in compliance with the License.
# 你可以在以下链接获取许可副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律适用或书面协议另有约定，否则软件在"按现状"基础上分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可以获取特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入数学模块以进行数学运算
import math
# 导入数据类装饰器以创建数据类
from dataclasses import dataclass
# 导入可选类型、元组和联合类型的类型提示
from typing import Optional, Tuple, Union

# 导入 NumPy 以进行数组操作
import numpy as np
# 导入 PyTorch 库以进行深度学习操作
import torch

# 从配置工具导入配置混合类和注册到配置的功能
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入基本输出类
from ..utils import BaseOutput
# 从 PyTorch 工具模块导入随机张量生成函数
from ..utils.torch_utils import randn_tensor
# 从调度工具模块导入调度混合类
from .scheduling_utils import SchedulerMixin


# 定义调度器输出的输出类，继承自基本输出类
@dataclass
class RePaintSchedulerOutput(BaseOutput):
    """
    调度器步进函数输出的输出类。

    参数:
        prev_sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)` 的图像):
            前一个时间步的计算样本 (x_{t-1})。 `prev_sample` 应作为下一个模型输入
            在去噪循环中使用。
        pred_original_sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)` 的图像):
            基于当前时间步模型输出的预测去噪样本 (x_{0})。 `pred_original_sample` 可用于预览进展或指导。
    """

    # 前一个时间步的样本
    prev_sample: torch.Tensor
    # 预测的去噪样本
    pred_original_sample: torch.Tensor


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建一个 beta 调度程序，离散化给定的 alpha_t_bar 函数，该函数定义了时间 t = [0,1] 的 (1-beta) 的累积乘积。

    包含一个 alpha_bar 函数，该函数接受参数 t 并将其转换为到当前扩散过程部分的 (1-beta) 的累积乘积。

    参数:
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta；使用小于 1 的值以
                     防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择

    返回:
        betas (`np.ndarray`): 调度程序用于模型输出步骤的 beta 值
    """
    # 如果选择了余弦作为变换类型
    if alpha_transform_type == "cosine":
        # 定义 alpha_bar 函数，基于余弦计算
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 如果选择了指数作为变换类型
    elif alpha_transform_type == "exp":
        # 定义 alpha_bar 函数，基于指数计算
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
    # 如果 alpha_transform_type 不受支持，则引发一个值错误，包含不支持的类型信息
        else:
            raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    
    # 初始化一个空列表，用于存储 beta 值
        betas = []
    # 遍历每一个扩散时间步长
        for i in range(num_diffusion_timesteps):
            # 计算当前时间步的比例 t1
            t1 = i / num_diffusion_timesteps
            # 计算下一个时间步的比例 t2
            t2 = (i + 1) / num_diffusion_timesteps
            # 计算 beta 值，确保不超过 max_beta，并添加到列表中
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 beta 列表转换为 PyTorch 张量，并指定数据类型为 float32
        return torch.tensor(betas, dtype=torch.float32)
# 定义一个名为 RePaintScheduler 的类，继承自 SchedulerMixin 和 ConfigMixin
class RePaintScheduler(SchedulerMixin, ConfigMixin):
    """
    `RePaintScheduler` 是用于在给定掩码内进行 DDPM 修复的调度器。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关库为所有调度器实现的通用
    方法（如加载和保存）的文档，请查看超类文档。

    参数：
        num_train_timesteps (`int`, defaults to 1000):
            训练模型的扩散步骤数量。
        beta_start (`float`, defaults to 0.0001):
            推理的起始 `beta` 值。
        beta_end (`float`, defaults to 0.02):
            最终的 `beta` 值。
        beta_schedule (`str`, defaults to `"linear"`):
            beta 调度，一个将 beta 范围映射到模型步进序列的映射。可选值有
            `linear`、`scaled_linear`、`squaredcos_cap_v2` 或 `sigmoid`。
        eta (`float`):
            在扩散步骤中添加噪声的噪声权重。如果值在 0.0 和 1.0 之间，对应于 DDIM 调度器；
            如果值在 -0.0 和 1.0 之间，对应于 DDPM 调度器。
        trained_betas (`np.ndarray`, *optional*):
            直接将 beta 数组传递给构造函数，以绕过 `beta_start` 和 `beta_end`。
        clip_sample (`bool`, defaults to `True`):
            将预测样本裁剪在 -1 和 1 之间，以确保数值稳定性。

    """

    # 定义调度器的顺序为 1
    order = 1

    # 注册到配置的方法，初始化调度器的参数
    @register_to_config
    def __init__(
        # 训练步骤数量，默认为 1000
        self,
        num_train_timesteps: int = 1000,
        # 起始 beta 值，默认为 0.0001
        beta_start: float = 0.0001,
        # 最终 beta 值，默认为 0.02
        beta_end: float = 0.02,
        # beta 调度类型，默认为 "linear"
        beta_schedule: str = "linear",
        # 噪声权重，默认为 0.0
        eta: float = 0.0,
        # 可选的 beta 数组，默认为 None
        trained_betas: Optional[np.ndarray] = None,
        # 是否裁剪样本，默认为 True
        clip_sample: bool = True,
    ):
        # 检查训练好的 beta 值是否为 None
        if trained_betas is not None:
            # 将训练好的 beta 值转换为 PyTorch 张量
            self.betas = torch.from_numpy(trained_betas)
        # 检查 beta 调度是否为线性
        elif beta_schedule == "linear":
            # 创建线性范围的 beta 值，从 beta_start 到 beta_end
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 检查 beta 调度是否为缩放线性
        elif beta_schedule == "scaled_linear":
            # 这个调度特定于潜在扩散模型
            # 创建 beta 值，先取平方根，再平方
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 检查 beta 调度是否为 squaredcos_cap_v2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            # 使用 alpha_bar 函数生成 beta 值
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 检查 beta 调度是否为 sigmoid
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid 调度
            # 创建范围从 -6 到 6 的线性 beta 值
            betas = torch.linspace(-6, 6, num_train_timesteps)
            # 使用 sigmoid 函数缩放 beta 值，并映射到 [beta_start, beta_end]
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            # 如果提供的 beta_schedule 未实现，抛出异常
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alphas，等于 1 减去 betas
        self.alphas = 1.0 - self.betas
        # 计算 alphas 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 创建一个值为 1.0 的张量
        self.one = torch.tensor(1.0)

        # 初始化最终的累积 alpha 值
        self.final_alpha_cumprod = torch.tensor(1.0)

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的值
        # 推理步骤数初始化为 None
        self.num_inference_steps = None
        # 创建一个张量，表示从 0 到 num_train_timesteps 的反向数组
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        # 初始化 eta 值
        self.eta = eta

    # 定义缩放模型输入的方法
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
        # 返回原样本，未进行缩放
        return sample

    # 定义设置时间步的方法
    def set_timesteps(
        self,
        num_inference_steps: int,
        jump_length: int = 10,
        jump_n_sample: int = 10,
        device: Union[str, torch.device] = None,
    # 定义用于设置离散时间步长的函数，在推理前运行
        ):
            """
            设置扩散链的离散时间步（在推理之前运行）。
    
            Args:
                num_inference_steps (`int`):
                    用于生成样本的扩散步骤数量，如果使用，则`timesteps`必须为`None`。
                jump_length (`int`, defaults to 10):
                    在时间上向前跳跃的步数，在进行一次跳跃时向后移动时间（在RePaint论文中表示为“j”）。请参阅论文中的图9和10。
                jump_n_sample (`int`, defaults to 10):
                    对于所选时间样本，进行向前时间跳跃的次数。请参阅论文中的图9和10。
                device (`str` or `torch.device`, *optional*):
                    应将时间步移动到的设备。如果为`None`，则时间步不移动。
    
            """
            # 选择最小值，确保推理步骤不超过训练时间步
            num_inference_steps = min(self.config.num_train_timesteps, num_inference_steps)
            # 将推理步骤数量保存到实例变量
            self.num_inference_steps = num_inference_steps
    
            # 初始化时间步列表
            timesteps = []
    
            # 创建跳跃字典以记录时间跳跃的信息
            jumps = {}
            # 根据给定的跳跃长度，计算每个时间步的跳跃数量
            for j in range(0, num_inference_steps - jump_length, jump_length):
                jumps[j] = jump_n_sample - 1
    
            # 初始化时间变量为推理步骤数量
            t = num_inference_steps
            # 从推理步骤开始，逐步向前计算时间步
            while t >= 1:
                # 递减时间步
                t = t - 1
                # 将当前时间步添加到列表
                timesteps.append(t)
    
                # 检查当前时间步是否需要跳跃
                if jumps.get(t, 0) > 0:
                    jumps[t] = jumps[t] - 1
                    # 对于每个跳跃长度，向前增加时间步
                    for _ in range(jump_length):
                        t = t + 1
                        # 将跳跃后的时间步添加到列表
                        timesteps.append(t)
    
            # 将时间步数组乘以缩放因子以适应训练时间步
            timesteps = np.array(timesteps) * (self.config.num_train_timesteps // self.num_inference_steps)
            # 将时间步转换为张量并移动到指定设备
            self.timesteps = torch.from_numpy(timesteps).to(device)
    
        # 定义获取方差的函数
        def _get_variance(self, t):
            # 计算前一个时间步
            prev_timestep = t - self.config.num_train_timesteps // self.num_inference_steps
    
            # 获取当前和前一个时间步的累积alpha值
            alpha_prod_t = self.alphas_cumprod[t]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
            # 计算当前和前一个时间步的beta值
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
    
            # 对于t > 0，计算预测方差，具体公式见论文中的公式（6）和（7）
            # 预测方差的计算公式，获取前一个样本
            variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    
            # 返回计算得到的方差
            return variance
    
        # 定义步骤函数，用于执行模型推理
        def step(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            original_image: torch.Tensor,
            mask: torch.Tensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
    # 定义撤销步骤的函数，接受样本、时间步和可选的生成器
    def undo_step(self, sample, timestep, generator=None):
        # 计算每个推断步骤的训练时间步数
        n = self.config.num_train_timesteps // self.num_inference_steps
    
        # 循环 n 次，进行每个时间步的处理
        for i in range(n):
            # 获取当前时间步对应的 beta 值
            beta = self.betas[timestep + i]
            # 如果设备类型是 MPS，则处理随机噪声
            if sample.device.type == "mps":
                # 在 MPS 上 randn 生成的随机数不具可复现性
                noise = randn_tensor(sample.shape, dtype=sample.dtype, generator=generator)
                # 将噪声移动到样本所在的设备
                noise = noise.to(sample.device)
            else:
                # 在其他设备上生成随机噪声
                noise = randn_tensor(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
    
            # 更新样本，按照公式进行噪声混合
            sample = (1 - beta) ** 0.5 * sample + beta**0.5 * noise
    
        # 返回更新后的样本
        return sample
    
    # 定义添加噪声的函数，接受原始样本、噪声和时间步，返回张量
    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.IntTensor,
        ) -> torch.Tensor:
        # 抛出未实现的错误，提示使用指定的方法进行训练
        raise NotImplementedError("Use `DDPMScheduler.add_noise()` to train for sampling with RePaint.")
    
    # 定义返回训练时间步数的函数
    def __len__(self):
        # 返回配置中定义的训练时间步数
        return self.config.num_train_timesteps
```