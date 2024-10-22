# `.\diffusers\schedulers\scheduling_flow_match_heun_discrete.py`

```py
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 你可以在遵循许可证的情况下使用此文件。
# 你可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是以“原样”基础提供的，
# 不提供任何形式的担保或条件，无论是明示或暗示的。
# 有关许可证的具体权限和限制，请参阅许可证。

from dataclasses import dataclass  # 导入数据类装饰器，用于创建简单的类
from typing import Optional, Tuple, Union  # 导入类型提示，Optional表示可选类型，Tuple和Union用于表示元组和联合类型

import numpy as np  # 导入NumPy库，通常用于数组操作
import torch  # 导入PyTorch库，通常用于深度学习

from ..configuration_utils import ConfigMixin, register_to_config  # 从上级模块导入配置混合类和注册配置的装饰器
from ..utils import BaseOutput, logging  # 从上级模块导入基础输出类和日志工具
from ..utils.torch_utils import randn_tensor  # 从上级模块导入用于生成随机张量的函数
from .scheduling_utils import SchedulerMixin  # 从当前模块导入调度混合类

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，命名为模块名称

@dataclass  # 使用数据类装饰器，自动生成初始化和其他方法
class FlowMatchHeunDiscreteSchedulerOutput(BaseOutput):  # 定义FlowMatchHeunDiscreteSchedulerOutput类，继承自BaseOutput
    """
    Output class for the scheduler's `step` function output.
    
    调度器`step`函数的输出类。

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor  # 定义一个属性prev_sample，类型为torch.FloatTensor，表示前一步的样本

class FlowMatchHeunDiscreteScheduler(SchedulerMixin, ConfigMixin):  # 定义FlowMatchHeunDiscreteScheduler类，继承自SchedulerMixin和ConfigMixin
    """
    Heun scheduler.

    这是一个Heun调度器。

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    该模型继承自[`SchedulerMixin`]和[`ConfigMixin`]。请查阅超类文档以获取库为所有调度器实现的通用方法，例如加载和保存。

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
            训练模型的扩散步数。
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
            时间步的缩放方式。有关更多信息，请参阅[常见扩散噪声调度和采样步骤存在缺陷]的表2。
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
            时间步调度的位移值。
    """

    _compatibles = []  # 定义一个类属性，用于存储兼容性信息，初始为空列表
    order = 2  # 定义一个类属性，表示调度器的顺序，初始值为2

    @register_to_config  # 使用装饰器将此方法注册到配置中
    def __init__(  # 定义初始化方法
        self,  # 引用自身
        num_train_timesteps: int = 1000,  # 定义num_train_timesteps参数，默认为1000，表示训练步数
        shift: float = 1.0,  # 定义shift参数，默认为1.0，表示时间步调度的位移值
    ):
        # 创建一个线性空间，范围从 1 到 num_train_timesteps，生成 num_train_timesteps 个点，并反转顺序
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        # 将 numpy 数组转换为 PyTorch 的张量，并指定数据类型为 float32
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        # 计算 sigmas，sigmas 为 timesteps 相对于 num_train_timesteps 的比例
        sigmas = timesteps / num_train_timesteps
        # 根据 shift 调整 sigmas 的值，进行非线性缩放
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # 将调整后的 sigmas 乘以 num_train_timesteps，保存为 self.timesteps
        self.timesteps = sigmas * num_train_timesteps

        # 初始化步索引为 None，表示当前未定义
        self._step_index = None
        # 初始化起始索引为 None，表示当前未定义
        self._begin_index = None

        # 将 sigmas 移动到 CPU，避免过多的 CPU/GPU 通信
        self.sigmas = sigmas.to("cpu")  
        # 记录 sigmas 的最小值，将其转为标量
        self.sigma_min = self.sigmas[-1].item()
        # 记录 sigmas 的最大值，将其转为标量
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后将增加 1。
        """
        # 返回当前的步索引
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过 `set_begin_index` 方法从管道设置。
        """
        # 返回当前的起始索引
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 中复制
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的起始索引。此函数应在推理前从管道运行。

        参数:
            begin_index (`int`):
                调度器的起始索引。
        """
        # 将起始索引设置为指定值
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        流匹配中的前向过程

        参数:
            sample (`torch.FloatTensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。

        返回:
            `torch.FloatTensor`:
                缩放后的输入样本。
        """
        # 如果当前步索引为 None，则初始化步索引
        if self.step_index is None:
            self._init_step_index(timestep)

        # 获取当前步索引对应的 sigma 值
        sigma = self.sigmas[self.step_index]
        # 使用 sigma 对噪声和样本进行加权组合，得到新的样本
        sample = sigma * noise + (1.0 - sigma) * sample

        # 返回处理后的样本
        return sample

    def _sigma_to_t(self, sigma):
        # 将 sigma 转换为与训练时间步数相关的值
        return sigma * self.config.num_train_timesteps
    # 设置离散时间步长，用于扩散链（在推断之前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步长（在推断之前运行）。
    
        参数：
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数量。
            device (`str` 或 `torch.device`, *可选*):
                时间步长应该移动到的设备。如果为 `None`，则时间步长不移动。
        """
        # 将输入的推断步骤数量保存到实例变量中
        self.num_inference_steps = num_inference_steps
    
        # 生成从最大 sigma 到最小 sigma 的均匀时间步长
        timesteps = np.linspace(
            self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
        )
    
        # 将时间步长归一化到训练时间步数
        sigmas = timesteps / self.config.num_train_timesteps
        # 根据配置的偏移量调整 sigma 值
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        # 将 numpy 数组转换为张量，并移动到指定设备
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
    
        # 计算最终的时间步长
        timesteps = sigmas * self.config.num_train_timesteps
        # 扩展时间步长，使其重复交替
        timesteps = torch.cat([timesteps[:1], timesteps[1:].repeat_interleave(2)])
        # 将时间步长保存到实例变量中，并移动到指定设备
        self.timesteps = timesteps.to(device=device)
    
        # 为 sigma 添加零值的张量
        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        # 扩展 sigma，使其重复交替
        self.sigmas = torch.cat([sigmas[:1], sigmas[1:-1].repeat_interleave(2), sigmas[-1:]])
    
        # 清空导数和时间增量
        self.prev_derivative = None
        self.dt = None
    
        # 初始化步长和起始索引
        self._step_index = None
        self._begin_index = None
    
    # 根据时间步长获取对应的索引
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果未提供调度时间步长，则使用实例中的时间步长
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
    
        # 找到与给定时间步长相等的索引
        indices = (schedule_timesteps == timestep).nonzero()
    
        # 选择第一步的 sigma 索引，避免跳过 sigma
        pos = 1 if len(indices) > 1 else 0
    
        # 返回对应索引的值
        return indices[pos].item()
    
    # 初始化步长索引
    def _init_step_index(self, timestep):
        # 如果起始索引为 None，则计算步长索引
        if self.begin_index is None:
            # 如果时间步长是张量，则转换到相应设备
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 根据时间步长获取当前步长索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则，使用已定义的起始索引
            self._step_index = self._begin_index
    
    # 判断是否为一阶状态
    @property
    def state_in_first_order(self):
        # 如果 dt 为 None，返回 True
        return self.dt is None
    
    # 执行一步更新
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        # 返回训练时间步数的数量
        def __len__(self):
            return self.config.num_train_timesteps
```