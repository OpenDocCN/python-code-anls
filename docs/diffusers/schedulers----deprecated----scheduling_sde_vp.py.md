# `.\diffusers\schedulers\deprecated\scheduling_sde_vp.py`

```py
# 版权声明，说明版权所有者及保留权利
# Copyright 2024 Google Brain and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版授权使用本文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵循许可证的情况下才能使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”提供，不附带任何形式的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以获取特定的权限和限制信息
# See the License for the specific language governing permissions and
# limitations under the License.

# 免责声明：此文件受到 https://github.com/yang-song/score_sde_pytorch 的强烈影响
# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

# 导入数学模块
import math
# 从 typing 导入 Union 类型
from typing import Union

# 导入 PyTorch 库
import torch

# 从配置实用程序导入 ConfigMixin 和注册配置装饰器
from ...configuration_utils import ConfigMixin, register_to_config
# 从 PyTorch 实用程序导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从调度实用程序导入调度器混合类
from ..scheduling_utils import SchedulerMixin

# 定义 ScoreSdeVpScheduler 类，继承自 SchedulerMixin 和 ConfigMixin
class ScoreSdeVpScheduler(SchedulerMixin, ConfigMixin):
    """
    `ScoreSdeVpScheduler` 是一种保方差的随机微分方程 (SDE) 调度器。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。查看超类文档以获取库为所有调度器实现的通用方法，例如加载和保存。

    参数：
        num_train_timesteps (`int`, defaults to 2000):
            训练模型的扩散步数。
        beta_min (`int`, defaults to 0.1):
        beta_max (`int`, defaults to 20):
        sampling_eps (`int`, defaults to 1e-3):
            采样结束值，时间步逐渐从 1 减少到 epsilon。
    """

    # 定义类的顺序属性
    order = 1

    # 注册构造函数到配置
    @register_to_config
    def __init__(self, num_train_timesteps=2000, beta_min=0.1, beta_max=20, sampling_eps=1e-3):
        # 初始化 sigmas 为 None
        self.sigmas = None
        # 初始化 discrete_sigmas 为 None
        self.discrete_sigmas = None
        # 初始化 timesteps 为 None
        self.timesteps = None

    # 定义设置时间步的方法
    def set_timesteps(self, num_inference_steps, device: Union[str, torch.device] = None):
        """
        设置扩散链所使用的连续时间步（在推理之前运行）。

        参数：
            num_inference_steps (`int`):
                生成样本时使用的扩散步数。
            device (`str` 或 `torch.device`, *可选*):
                要移动时间步的设备。如果为 `None`，则时间步不会移动。
        """
        # 生成从 1 到 sampling_eps 的线性空间，数量为 num_inference_steps，指定设备
        self.timesteps = torch.linspace(1, self.config.sampling_eps, num_inference_steps, device=device)
    # 定义一个步骤预测函数，通过逆向 SDE 从前一时间步预测样本
    def step_pred(self, score, x, t, generator=None):
        """
        从之前的时间步预测样本，通过逆向 SDE。该函数从学习的模型输出（通常是预测的噪声）传播扩散过程。

        参数：
            score (): 模型输出的分数
            x (): 当前样本
            t (): 当前时间步
            generator (`torch.Generator`, *可选*): 随机数生成器
        """
        # 检查 timesteps 是否被设置
        if self.timesteps is None:
            # 如果未设置 timesteps，抛出错误
            raise ValueError(
                "`self.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # TODO(Patrick) 更好的注释 + 非 PyTorch 处理
        # 后处理模型得分
        log_mean_coeff = -0.25 * t**2 * (self.config.beta_max - self.config.beta_min) - 0.5 * t * self.config.beta_min
        # 计算标准差，使用对数均值系数
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        # 将标准差展平为一维
        std = std.flatten()
        # 使标准差的维度与得分的维度匹配
        while len(std.shape) < len(score.shape):
            std = std.unsqueeze(-1)
        # 对得分进行标准化处理
        score = -score / std

        # 计算时间增量
        dt = -1.0 / len(self.timesteps)

        # 计算当前时间步的 beta 值
        beta_t = self.config.beta_min + t * (self.config.beta_max - self.config.beta_min)
        # 将 beta_t 展平为一维
        beta_t = beta_t.flatten()
        # 使 beta_t 的维度与样本 x 的维度匹配
        while len(beta_t.shape) < len(x.shape):
            beta_t = beta_t.unsqueeze(-1)
        # 计算漂移项
        drift = -0.5 * beta_t * x

        # 计算扩散项
        diffusion = torch.sqrt(beta_t)
        # 更新漂移项，包含模型得分的影响
        drift = drift - diffusion**2 * score
        # 计算 x 的均值
        x_mean = x + drift * dt

        # 添加噪声
        noise = randn_tensor(x.shape, layout=x.layout, generator=generator, device=x.device, dtype=x.dtype)
        # 更新样本 x，加入扩散项和噪声
        x = x_mean + diffusion * math.sqrt(-dt) * noise

        # 返回更新后的样本和均值
        return x, x_mean

    # 定义 __len__ 方法，返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```