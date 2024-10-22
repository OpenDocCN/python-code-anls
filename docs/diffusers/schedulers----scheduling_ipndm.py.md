# `.\diffusers\schedulers\scheduling_ipndm.py`

```py
# 版权信息，声明版权所有者及相关许可信息
# Copyright 2024 Zhejiang University Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 许可本文件；
# 仅在遵守许可证的情况下使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 在许可证下按“原样”分发，不提供任何形式的保证或条件，
# 明示或暗示。
# 参见许可证以获取特定语言关于权限和
# 限制的信息。

# 导入数学库
import math
# 从类型提示库导入所需类型
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch

# 从配置工具中导入 ConfigMixin 和注册到配置的装饰器
from ..configuration_utils import ConfigMixin, register_to_config
# 从调度工具中导入 SchedulerMixin 和 SchedulerOutput
from .scheduling_utils import SchedulerMixin, SchedulerOutput


class IPNDMScheduler(SchedulerMixin, ConfigMixin):
    """
    一个四阶改进伪线性多步调度器。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。查看父类文档以了解库为所有调度器实现的通用
    方法，例如加载和保存。

    参数：
        num_train_timesteps (`int`, 默认为 1000):
            用于训练模型的扩散步骤数量。
        trained_betas (`np.ndarray`, *可选*):
            直接传递一组 beta 数组到构造函数，以绕过 `beta_start` 和 `beta_end`。
    """

    # 调度器的阶数设置为 1
    order = 1

    # 注册到配置的方法
    @register_to_config
    def __init__(
        # 设置训练的扩散时间步数，默认为 1000
        self, num_train_timesteps: int = 1000, trained_betas: Optional[Union[np.ndarray, List[float]]] = None
    ):
        # 设置 `betas`，`alphas` 和 `timesteps`
        self.set_timesteps(num_train_timesteps)

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 当前仅支持 F-PNDM，即 Runge-Kutta 方法
        # 有关算法的更多信息，请查看论文：https://arxiv.org/pdf/2202.09778.pdf
        # 主要参考公式 (9)，(12)，(13) 和算法 2。
        self.pndm_order = 4

        # 运行时值的列表
        self.ets = []
        # 当前步骤索引初始化为 None
        self._step_index = None
        # 开始索引初始化为 None
        self._begin_index = None

    # 属性，获取当前时间步的索引
    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加 1。
        """
        return self._step_index

    # 属性，获取第一个时间步的索引
    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过 `set_begin_index` 方法从管道设置。
        """
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler 复制的方法 set_begin_index
    # 设置调度器的开始索引，默认值为 0
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的开始索引。此函数应在推理前从管道中运行。

        Args:
            begin_index (`int`):
                调度器的开始索引。
        """
        # 将传入的开始索引保存到实例变量中
        self._begin_index = begin_index

    # 设置用于扩散链的离散时间步（在推理前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步（在推理前运行）。

        Args:
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数。
            device (`str` or `torch.device`, *optional*):
                时间步应移动到的设备。如果为 `None`，则不移动时间步。
        """
        # 保存扩散步骤数到实例变量中
        self.num_inference_steps = num_inference_steps
        # 创建一个从 1 到 0 的线性空间，步数为 num_inference_steps + 1，去掉最后一个元素
        steps = torch.linspace(1, 0, num_inference_steps + 1)[:-1]
        # 在时间步的末尾添加一个 0.0
        steps = torch.cat([steps, torch.tensor([0.0])])

        # 如果训练的 beta 值存在，将其转换为张量
        if self.config.trained_betas is not None:
            self.betas = torch.tensor(self.config.trained_betas, dtype=torch.float32)
        else:
            # 否则根据步骤计算 beta 值
            self.betas = torch.sin(steps * math.pi / 2) ** 2

        # 计算 alpha 值，作为 beta 值的平方根
        self.alphas = (1.0 - self.betas**2) ** 0.5

        # 计算时间步的角度值并去掉最后一个元素
        timesteps = (torch.atan2(self.betas, self.alphas) / math.pi * 2)[:-1]
        # 将时间步移到指定设备上
        self.timesteps = timesteps.to(device)

        # 初始化空列表以存储 ets
        self.ets = []
        # 初始化步索引和开始索引为 None
        self._step_index = None
        self._begin_index = None

    # 从 EulerDiscreteScheduler 复制的函数，获取给定时间步的索引
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果没有提供时间步，则使用实例中的时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与给定时间步相等的索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 确保第一个步骤的 sigma 索引总是第二个索引（或唯一一个时的最后一个索引）
        pos = 1 if len(indices) > 1 else 0

        # 返回找到的索引值
        return indices[pos].item()

    # 从 EulerDiscreteScheduler 复制的函数，初始化步索引
    def _init_step_index(self, timestep):
        # 如果开始索引为 None
        if self.begin_index is None:
            # 将时间步转换为与时间步设备相同的张量
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 根据时间步设置步索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则，将步索引设置为开始索引
            self._step_index = self._begin_index

    # 步骤函数，接收模型输出和样本进行处理
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        return_dict: bool = True,
    # 定义函数，返回调度器输出或元组
    ) -> Union[SchedulerOutput, Tuple]:
        # 预测从上一个时间步生成的样本，通过反向 SDE 进行传播
        """
        # 文档字符串，描述函数的目的和参数
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the linear multistep method. It performs one forward pass multiple times to approximate the solution.
    
        Args:
            model_output (`torch.Tensor`):
                从学习的扩散模型直接输出的张量。
            timestep (`int`):
                当前扩散链中的离散时间步。
            sample (`torch.Tensor`):
                通过扩散过程创建的当前样本实例。
            return_dict (`bool`):
                是否返回一个 [`~schedulers.scheduling_utils.SchedulerOutput`] 或元组。
    
        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_utils.SchedulerOutput`]，否则返回一个
                元组，其中第一个元素是样本张量。
        """
        # 检查推理步骤是否为 None，若是则抛出错误
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        # 如果步索引为 None，初始化步索引
        if self.step_index is None:
            self._init_step_index(timestep)
    
        # 当前时间步索引
        timestep_index = self.step_index
        # 前一个时间步索引
        prev_timestep_index = self.step_index + 1
    
        # 根据当前样本和模型输出计算 ets
        ets = sample * self.betas[timestep_index] + model_output * self.alphas[timestep_index]
        # 将计算得到的 ets 添加到列表中
        self.ets.append(ets)
    
        # 根据 ets 的长度进行不同的线性多步法计算
        if len(self.ets) == 1:
            ets = self.ets[-1]
        elif len(self.ets) == 2:
            ets = (3 * self.ets[-1] - self.ets[-2]) / 2
        elif len(self.ets) == 3:
            ets = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        else:
            ets = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])
    
        # 获取前一个样本
        prev_sample = self._get_prev_sample(sample, timestep_index, prev_timestep_index, ets)
    
        # 完成后步索引加一
        self._step_index += 1
    
        # 根据 return_dict 决定返回格式
        if not return_dict:
            return (prev_sample,)
    
        # 返回调度器输出
        return SchedulerOutput(prev_sample=prev_sample)
    
    # 定义函数，确保模型输入的可互换性
    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 文档字符串，描述函数的目的和参数
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
    
        Args:
            sample (`torch.Tensor`):
                输入样本。
    
        Returns:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 返回未缩放的输入样本
        return sample
    # 定义一个私有方法，用于获取前一个样本
        def _get_prev_sample(self, sample, timestep_index, prev_timestep_index, ets):
            # 获取当前时间步的 alpha 值
            alpha = self.alphas[timestep_index]
            # 获取当前时间步的 sigma 值
            sigma = self.betas[timestep_index]
    
            # 获取前一个时间步的 alpha 值
            next_alpha = self.alphas[prev_timestep_index]
            # 获取前一个时间步的 sigma 值
            next_sigma = self.betas[prev_timestep_index]
    
            # 根据当前样本、sigma 和 ets 计算预测值
            pred = (sample - sigma * ets) / max(alpha, 1e-8)
            # 通过预测值、前一个时间步的 alpha 和 sigma 计算前一个样本
            prev_sample = next_alpha * pred + ets * next_sigma
    
            # 返回计算得到的前一个样本
            return prev_sample
    
        # 定义一个方法，用于获取训练时间步的数量
        def __len__(self):
            # 返回配置中训练时间步的数量
            return self.config.num_train_timesteps
```