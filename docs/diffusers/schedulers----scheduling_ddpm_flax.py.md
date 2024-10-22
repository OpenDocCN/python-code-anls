# `.\diffusers\schedulers\scheduling_ddpm_flax.py`

```py
# 版权声明，标明文件的版权归属
# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证授权，此处说明用户使用该文件的条款
# Licensed under the Apache License, Version 2.0 (the "License");
# 只有在遵循许可证的情况下才能使用该文件
# you may not use this file except in compliance with the License.
# 许可证可以在以下网址获取
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 本文件在适用法律或书面协议下是“按现状”提供的，不附带任何形式的保证或条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 具体的许可证条款和条件可以在下方找到
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 本文件受到 https://github.com/ermongroup/ddim 的强烈影响
# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

# 导入数据类功能，用于定义简单的类
from dataclasses import dataclass
# 导入类型定义，便于类型注解
from typing import Optional, Tuple, Union

# 导入 Flax 和 JAX 库，用于机器学习
import flax
import jax
import jax.numpy as jnp

# 导入配置和调度相关的工具
from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
    # 导入通用调度器状态
    CommonSchedulerState,
    # 导入 Karras 扩散调度器
    FlaxKarrasDiffusionSchedulers,
    # 导入调度混合类
    FlaxSchedulerMixin,
    # 导入调度输出类
    FlaxSchedulerOutput,
    # 导入添加噪声的通用函数
    add_noise_common,
    # 导入获取速度的通用函数
    get_velocity_common,
)

# 定义 DDPMSchedulerState 数据类，保存调度器的状态信息
@flax.struct.dataclass
class DDPMSchedulerState:
    # 包含通用调度器状态
    common: CommonSchedulerState

    # 可设置的属性值
    init_noise_sigma: jnp.ndarray  # 初始噪声标准差
    timesteps: jnp.ndarray  # 时间步数组
    num_inference_steps: Optional[int] = None  # 推断步骤数，可选

    # 类方法，用于创建 DDPMSchedulerState 实例
    @classmethod
    def create(cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray):
        # 返回一个新的 DDPMSchedulerState 实例
        return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=timesteps)

# 定义 FlaxDDPMSchedulerOutput 数据类，继承自 FlaxSchedulerOutput
@dataclass
class FlaxDDPMSchedulerOutput(FlaxSchedulerOutput):
    state: DDPMSchedulerState  # 调度器的状态

# 定义 FlaxDDPMScheduler 类，结合调度和配置功能
class FlaxDDPMScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    Denoising diffusion probabilistic models (DDPMs) explores the connections between denoising score matching and
    Langevin dynamics sampling.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2006.11239
    """
    # 文档字符串，说明初始化函数的参数
    Args:
        num_train_timesteps (`int`): 训练模型所用的扩散步骤数量。
        beta_start (`float`): 推断的起始 `beta` 值。
        beta_end (`float`): 最终的 `beta` 值。
        beta_schedule (`str`):
            beta 调度，表示从 beta 范围到模型步骤序列的映射。可选值有
            `linear`、`scaled_linear` 或 `squaredcos_cap_v2`。
        trained_betas (`np.ndarray`, optional):
            直接传递 beta 数组到构造函数的选项，以绕过 `beta_start`、`beta_end` 等参数。
        variance_type (`str`):
            用于添加噪声到去噪样本时裁剪方差的选项。可选值有 `fixed_small`、
            `fixed_small_log`、`fixed_large`、`fixed_large_log`、`learned` 或 `learned_range`。
        clip_sample (`bool`, default `True`):
            裁剪预测样本在 -1 和 1 之间以确保数值稳定性的选项。
        prediction_type (`str`, default `epsilon`):
            指示模型是预测噪声（epsilon）还是样本。可选值为 `epsilon`、`sample`。
            对于此调度器不支持 `v-prediction`。
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            用于参数和计算的 `dtype` 类型。
    """

    # 获取所有兼容的调度器名称
    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    # 定义数据类型的变量
    dtype: jnp.dtype

    # 属性，返回是否有状态
    @property
    def has_state(self):
        # 返回 True，表示有状态信息
        return True

    # 注册到配置的初始化方法
    @register_to_config
    def __init__(
        # 训练步骤的默认值
        num_train_timesteps: int = 1000,
        # 起始 beta 值的默认值
        beta_start: float = 0.0001,
        # 最终 beta 值的默认值
        beta_end: float = 0.02,
        # beta 调度的默认值
        beta_schedule: str = "linear",
        # 训练好的 beta 数组，默认为 None
        trained_betas: Optional[jnp.ndarray] = None,
        # 方差类型的默认值
        variance_type: str = "fixed_small",
        # 裁剪样本的默认值
        clip_sample: bool = True,
        # 预测类型的默认值
        prediction_type: str = "epsilon",
        # 数据类型的默认值
        dtype: jnp.dtype = jnp.float32,
    ):
        # 设置数据类型
        self.dtype = dtype

    # 创建状态的方法
    def create_state(self, common: Optional[CommonSchedulerState] = None) -> DDPMSchedulerState:
        # 如果未提供公共状态，则创建新的公共状态
        if common is None:
            common = CommonSchedulerState.create(self)

        # 初始噪声分布的标准差
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)

        # 生成从 num_train_timesteps 到 0 的时间步数组，进行反转
        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]

        # 返回创建的 DDPMSchedulerState 对象
        return DDPMSchedulerState.create(
            common=common,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )

    # 缩放模型输入的方法
    def scale_model_input(
        self, state: DDPMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
    ) -> jnp.ndarray:
        """
        Args:
            state (`PNDMSchedulerState`): `FlaxPNDMScheduler` 状态数据类实例。
            sample (`jnp.ndarray`): 输入样本
            timestep (`int`, optional): 当前时间步

        Returns:
            `jnp.ndarray`: 缩放后的输入样本
        """
        # 返回输入样本，未进行缩放处理
        return sample
    # 定义设置时间步长的方法，参数包括状态、推理步骤数量和形状
        def set_timesteps(
            self, state: DDPMSchedulerState, num_inference_steps: int, shape: Tuple = ()
        ) -> DDPMSchedulerState:
            """
            设置扩散链中使用的离散时间步长。用于推理前的辅助函数。
    
            Args:
                state (`DDIMSchedulerState`):
                    `FlaxDDPMScheduler`状态数据类实例。
                num_inference_steps (`int`):
                    在使用预训练模型生成样本时使用的扩散步骤数量。
            """
    
            # 计算步骤比例，通过训练时间步数除以推理步骤数量
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            # 通过乘以比例创建整数时间步，进行四舍五入以避免当推理步骤为3的幂时出现问题
            timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round()[::-1]
    
            # 返回替换后的状态，包括推理步骤数量和时间步
            return state.replace(
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
            )
    
        # 定义获取方差的方法，参数包括状态、时间步、预测方差和方差类型
        def _get_variance(self, state: DDPMSchedulerState, t, predicted_variance=None, variance_type=None):
            # 从状态中获取当前时间步的累积 alpha 值
            alpha_prod_t = state.common.alphas_cumprod[t]
            # 获取前一个时间步的累积 alpha 值，如果 t 大于 0，则取前一个值，否则为 1.0
            alpha_prod_t_prev = jnp.where(t > 0, state.common.alphas_cumprod[t - 1], jnp.array(1.0, dtype=self.dtype))
    
            # 计算预测方差 βt（见公式 (6) 和 (7)），并从中采样以获得前一个样本
            # x_{t-1} ~ N(pred_prev_sample, variance) == 将方差添加到预测样本中
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * state.common.betas[t]
    
            # 如果没有提供方差类型，则使用配置中的默认方差类型
            if variance_type is None:
                variance_type = self.config.variance_type
    
            # 进行各种处理以提高训练稳定性
            if variance_type == "fixed_small":
                # 将方差裁剪到最小值 1e-20
                variance = jnp.clip(variance, a_min=1e-20)
            # 对于 rl-diffuser，进行 log 转换处理
            elif variance_type == "fixed_small_log":
                variance = jnp.log(jnp.clip(variance, a_min=1e-20))
            elif variance_type == "fixed_large":
                # 使用当前时间步的 beta 值作为方差
                variance = state.common.betas[t]
            elif variance_type == "fixed_large_log":
                # 对当前时间步的 beta 值进行 log 转换
                variance = jnp.log(state.common.betas[t])
            elif variance_type == "learned":
                # 返回预测的方差
                return predicted_variance
            elif variance_type == "learned_range":
                # 计算最小和最大 log 方差
                min_log = variance
                max_log = state.common.betas[t]
                # 计算比例并混合方差
                frac = (predicted_variance + 1) / 2
                variance = frac * max_log + (1 - frac) * min_log
    
            # 返回计算后的方差
            return variance
    
        # 定义执行一步推理的步骤，参数包括状态、模型输出、时间步、样本和其他可选参数
        def step(
            self,
            state: DDPMSchedulerState,
            model_output: jnp.ndarray,
            timestep: int,
            sample: jnp.ndarray,
            key: Optional[jax.Array] = None,
            return_dict: bool = True,
        def add_noise(
            self,
            state: DDPMSchedulerState,
            original_samples: jnp.ndarray,
            noise: jnp.ndarray,
            timesteps: jnp.ndarray,
    # 返回添加噪声后的样本，基于公共状态和输入参数
        ) -> jnp.ndarray:
            return add_noise_common(state.common, original_samples, noise, timesteps)
    
    # 获取样本的速度，使用调度器状态和噪声等参数
        def get_velocity(
            self,
            state: DDPMSchedulerState,
            sample: jnp.ndarray,
            noise: jnp.ndarray,
            timesteps: jnp.ndarray,
        ) -> jnp.ndarray:
            # 调用公共函数以计算速度
            return get_velocity_common(state.common, sample, noise, timesteps)
    
    # 返回训练时间步数的数量
        def __len__(self):
            return self.config.num_train_timesteps
```