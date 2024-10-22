# `.\diffusers\schedulers\scheduling_sde_ve_flax.py`

```py
# 版权声明，表示文件的所有权和使用限制
# Copyright 2024 Google Brain and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证 2.0 版本的条款进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 只能在遵循许可证的前提下使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则此文件下的所有软件
# Unless required by applicable law or agreed to in writing, software
# 以“原样”方式分发，没有任何明示或暗示的担保或条件
# distributed under the License is distributed on an "AS IS" BASIS,
# 查看许可证以获取有关权限和限制的具体条款
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 声明：此文件受到 https://github.com/yang-song/score_sde_pytorch 的强烈影响
# DISCLAIMER: This file is strongly influenced by https://github.com/yang-song/score_sde_pytorch

# 导入所需模块
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import flax  # 导入 flax 用于构建可训练的模型
import jax  # 导入 jax 进行高效的数值计算
import jax.numpy as jnp  # 导入 jax 的 numpy 模块
from jax import random  # 导入 jax 的随机数生成模块

# 导入配置工具和调度器相关的功能
from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import FlaxSchedulerMixin, FlaxSchedulerOutput, broadcast_to_shape_from_left


@flax.struct.dataclass
class ScoreSdeVeSchedulerState:
    # 定义可设置的状态变量
    timesteps: Optional[jnp.ndarray] = None  # 时间步，默认为 None
    discrete_sigmas: Optional[jnp.ndarray] = None  # 离散 sigma 值，默认为 None
    sigmas: Optional[jnp.ndarray] = None  # sigma 值，默认为 None

    @classmethod
    def create(cls):
        # 创建并返回类的实例
        return cls()


@dataclass
class FlaxSdeVeOutput(FlaxSchedulerOutput):
    """
    ScoreSdeVeScheduler 的步骤函数输出的输出类。

    参数：
        state (`ScoreSdeVeSchedulerState`):
        prev_sample (`jnp.ndarray`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            计算的前一时间步的样本 (x_{t-1})。`prev_sample` 应作为下一次模型输入
            在去噪循环中使用。
        prev_sample_mean (`jnp.ndarray`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            平均的 `prev_sample`。与 `prev_sample` 相同，仅在之前的时间步上进行了均值平均。
    """

    state: ScoreSdeVeSchedulerState  # 调度器状态
    prev_sample: jnp.ndarray  # 前一时间步的样本
    prev_sample_mean: Optional[jnp.ndarray] = None  # 可选的前一样本均值


class FlaxScoreSdeVeScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    方差爆炸随机微分方程 (SDE) 调度器。

    有关更多信息，请参见原始论文：https://arxiv.org/abs/2011.13456

    [`~ConfigMixin`] 处理在调度器的 `__init__` 函数中传递的所有配置属性的存储
    例如 `num_train_timesteps`。可以通过 `scheduler.config.num_train_timesteps` 访问。
    [`SchedulerMixin`] 提供通过 [`SchedulerMixin.save_pretrained`] 和
    [`~SchedulerMixin.from_pretrained`] 函数进行通用加载和保存的功能。
    """
    # 函数参数说明，描述每个参数的意义和用途
    Args:
        num_train_timesteps (`int`): 模型训练时使用的扩散步骤数量
        snr (`float`): 
            权重系数，影响模型输出样本与随机噪声之间的关系
        sigma_min (`float`):
                采样过程中 sigma 序列的初始噪声规模，最小 sigma 应该与数据的分布相符
        sigma_max (`float`): 用于传入模型的连续时间步的最大值
        sampling_eps (`float`): 采样的结束值，时间步从 1 渐进减少到 epsilon
        correct_steps (`int`): 对生成样本执行的纠正步骤数量
    """

    # 定义一个属性，指示是否具有状态
    @property
    def has_state(self):
        return True

    # 初始化方法，设置模型的参数
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 2000,  # 训练步骤数量，默认为2000
        snr: float = 0.15,                 # 信噪比，默认为0.15
        sigma_min: float = 0.01,          # 最小噪声尺度，默认为0.01
        sigma_max: float = 1348.0,        # 最大噪声尺度，默认为1348.0
        sampling_eps: float = 1e-5,       # 采样结束值，默认为1e-5
        correct_steps: int = 1,            # 纠正步骤数量，默认为1
    ):
        pass  # 初始化方法为空，不执行任何操作

    # 创建状态方法，用于初始化调度器状态
    def create_state(self):
        # 创建调度器状态实例
        state = ScoreSdeVeSchedulerState.create()
        # 设置 sigma 值并返回更新后的状态
        return self.set_sigmas(
            state,
            self.config.num_train_timesteps,  # 使用配置中的训练步骤数量
            self.config.sigma_min,              # 使用配置中的最小 sigma
            self.config.sigma_max,              # 使用配置中的最大 sigma
            self.config.sampling_eps,           # 使用配置中的采样结束值
        )

    # 设置时间步的方法，更新状态以支持推理过程
    def set_timesteps(
        self, state: ScoreSdeVeSchedulerState, num_inference_steps: int, shape: Tuple = (), sampling_eps: float = None
    ) -> ScoreSdeVeSchedulerState:
        """
        设置扩散链中使用的连续时间步，推理前需运行的支持函数。

        Args:
            state (`ScoreSdeVeSchedulerState`): `FlaxScoreSdeVeScheduler` 的状态数据类实例
            num_inference_steps (`int`): 
                生成样本时使用的扩散步骤数量
            sampling_eps (`float`, optional): 
                最终时间步值（覆盖调度器实例化时给定的值）。

        """
        # 如果提供了采样结束值，则使用它，否则使用配置中的值
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps

        # 生成从1到采样结束值的均匀分布的时间步
        timesteps = jnp.linspace(1, sampling_eps, num_inference_steps)
        # 返回更新后的状态，包括新设置的时间步
        return state.replace(timesteps=timesteps)

    # 设置 sigma 值的方法，用于调整状态
    def set_sigmas(
        self,
        state: ScoreSdeVeSchedulerState,    # 当前状态实例
        num_inference_steps: int,           # 推理步骤数量
        sigma_min: float = None,            # 可选的最小 sigma 值
        sigma_max: float = None,            # 可选的最大 sigma 值
        sampling_eps: float = None,         # 可选的采样结束值
    ) -> ScoreSdeVeSchedulerState:
        """
        设置扩散链使用的噪声尺度。推理前运行的辅助函数。

        sigmas 控制样本更新中的 `drift` 和 `diffusion` 组件的权重。

        Args:
            state (`ScoreSdeVeSchedulerState`): `FlaxScoreSdeVeScheduler` 的状态数据类实例。
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数量，基于预训练模型。
            sigma_min (`float`, optional):
                初始噪声尺度值（覆盖在 Scheduler 实例化时给定的值）。
            sigma_max (`float`, optional):
                最终噪声尺度值（覆盖在 Scheduler 实例化时给定的值）。
            sampling_eps (`float`, optional):
                最终时间步值（覆盖在 Scheduler 实例化时给定的值）。
        """
        # 如果 sigma_min 为 None，则使用配置中的 sigma_min
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        # 如果 sigma_max 为 None，则使用配置中的 sigma_max
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        # 如果 sampling_eps 为 None，则使用配置中的 sampling_eps
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        # 如果 state 的时间步为 None，则设置时间步
        if state.timesteps is None:
            state = self.set_timesteps(state, num_inference_steps, sampling_eps)

        # 计算离散 sigma 的指数值，生成 num_inference_steps 个点
        discrete_sigmas = jnp.exp(jnp.linspace(jnp.log(sigma_min), jnp.log(sigma_max), num_inference_steps))
        # 计算当前时间步对应的 sigmas 数组
        sigmas = jnp.array([sigma_min * (sigma_max / sigma_min) ** t for t in state.timesteps])

        # 用离散 sigma 和 sigmas 替换状态中的相应值
        return state.replace(discrete_sigmas=discrete_sigmas, sigmas=sigmas)

    def get_adjacent_sigma(self, state, timesteps, t):
        # 返回与当前时间步相邻的 sigma 值，如果时间步为 0，则返回与之相同形状的零数组
        return jnp.where(timesteps == 0, jnp.zeros_like(t), state.discrete_sigmas[timesteps - 1])

    def step_pred(
        self,
        state: ScoreSdeVeSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        key: jax.Array,
        return_dict: bool = True,
    ) -> Union[FlaxSdeVeOutput, Tuple]:
        """
        预测在前一个时间步的样本，通过反向 SDE 实现。核心功能是从学习的模型输出（通常是预测的噪声）传播扩散过程。

        参数:
            state (`ScoreSdeVeSchedulerState`): `FlaxScoreSdeVeScheduler` 状态数据类实例。
            model_output (`jnp.ndarray`): 来自学习的扩散模型的直接输出。
            timestep (`int`): 扩散链中的当前离散时间步。
            sample (`jnp.ndarray`):
                当前正在通过扩散过程创建的样本实例。
            generator: 随机数生成器。
            return_dict (`bool`): 选项，决定返回元组还是 `FlaxSdeVeOutput` 类。

        返回:
            [`FlaxSdeVeOutput`] 或 `tuple`: 如果 `return_dict` 为 True，则返回 [`FlaxSdeVeOutput`]，否则返回一个元组。当返回元组时，第一个元素是样本张量。

        """
        # 检查状态的时间步是否设置，如果没有则抛出错误
        if state.timesteps is None:
            raise ValueError(
                "`state.timesteps` is not set, you need to run 'set_timesteps' after creating the scheduler"
            )

        # 将当前时间步扩展到与样本数量相同的形状
        timestep = timestep * jnp.ones(
            sample.shape[0],
        )
        # 计算离散时间步的索引
        timesteps = (timestep * (len(state.timesteps) - 1)).long()

        # 获取当前时间步对应的 sigma 值
        sigma = state.discrete_sigmas[timesteps]
        # 获取与当前时间步相邻的 sigma 值
        adjacent_sigma = self.get_adjacent_sigma(state, timesteps, timestep)
        # 初始化漂移项为与样本形状相同的零张量
        drift = jnp.zeros_like(sample)
        # 计算扩散项，表示噪声的变化
        diffusion = (sigma**2 - adjacent_sigma**2) ** 0.5

        # 根据论文中的公式 6，模型输出由网络建模为 grad_x log pt(x)
        # 公式 47 显示了 SDE 模型与祖先采样方法的类比
        diffusion = diffusion.flatten()  # 将扩散项展平为一维
        diffusion = broadcast_to_shape_from_left(diffusion, sample.shape)  # 将扩散项广播到样本形状
        drift = drift - diffusion**2 * model_output  # 更新漂移项

        # 根据扩散项采样噪声
        key = random.split(key, num=1)  # 分裂随机数生成器以获得新密钥
        noise = random.normal(key=key, shape=sample.shape)  # 生成与样本相同形状的随机噪声
        prev_sample_mean = sample - drift  # 计算前一个样本的均值，减去漂移项
        # TODO 检查变量 diffusion 是否为噪声的正确缩放项
        prev_sample = prev_sample_mean + diffusion * noise  # 加上扩散场对噪声的影响

        # 如果不返回字典，则返回一个元组，包含前一个样本、前一个样本均值和状态
        if not return_dict:
            return (prev_sample, prev_sample_mean, state)

        # 返回 FlaxSdeVeOutput 对象，包含前一个样本、前一个样本均值和状态
        return FlaxSdeVeOutput(prev_sample=prev_sample, prev_sample_mean=prev_sample_mean, state=state)

    def step_correct(
        self,
        state: ScoreSdeVeSchedulerState,
        model_output: jnp.ndarray,
        sample: jnp.ndarray,
        key: jax.Array,
        return_dict: bool = True,
    ) -> Union[FlaxSdeVeOutput, Tuple]:
        """
        根据网络输出的 model_output 校正预测样本，通常在每个时间步后反复执行。

        参数：
            state (`ScoreSdeVeSchedulerState`): `FlaxScoreSdeVeScheduler` 状态数据类实例。
            model_output (`jnp.ndarray`): 从学习的扩散模型直接输出。
            sample (`jnp.ndarray`):
                当前由扩散过程创建的样本实例。
            generator: 随机数生成器。
            return_dict (`bool`): 选项，用于返回元组而不是 FlaxSdeVeOutput 类。

        返回：
            [`FlaxSdeVeOutput`] 或 `tuple`: 如果 `return_dict` 为 True 则返回 [`FlaxSdeVeOutput`]，否则返回 `tuple`。返回元组时，第一个元素是样本张量。

        """
        # 检查 timesteps 是否被设置，如果未设置则抛出错误
        if state.timesteps is None:
            raise ValueError(
                "`state.timesteps` 未设置，需要在创建调度器后运行 'set_timesteps'"
            )

        # 对于小批量大小，论文建议用 sqrt(d) 替代 norm(z)，其中 d 是 z 的维度
        # 为校正生成噪声
        key = random.split(key, num=1)  # 将随机数生成器的 key 拆分
        noise = random.normal(key=key, shape=sample.shape)  # 生成与样本相同形状的噪声

        # 从 model_output、噪声和 snr 计算步长
        grad_norm = jnp.linalg.norm(model_output)  # 计算 model_output 的范数
        noise_norm = jnp.linalg.norm(noise)  # 计算噪声的范数
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2  # 计算步长
        step_size = step_size * jnp.ones(sample.shape[0])  # 扩展步长到样本批次大小

        # 计算校正样本：model_output 项和噪声项
        step_size = step_size.flatten()  # 将步长展平
        step_size = broadcast_to_shape_from_left(step_size, sample.shape)  # 将步长广播到样本形状
        prev_sample_mean = sample + step_size * model_output  # 计算校正样本均值
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise  # 添加噪声得到最终样本

        # 根据 return_dict 的值决定返回格式
        if not return_dict:
            return (prev_sample, state)  # 返回元组形式的样本和状态

        return FlaxSdeVeOutput(prev_sample=prev_sample, state=state)  # 返回 FlaxSdeVeOutput 对象

    def __len__(self):
        return self.config.num_train_timesteps  # 返回训练时间步的数量
```