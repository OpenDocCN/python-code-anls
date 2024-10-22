# `.\diffusers\schedulers\scheduling_euler_discrete_flax.py`

```py
# 版权所有 2024 Katherine Crowson 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）许可；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件在许可证下分发是按“原样”基础进行的，
# 不提供任何形式的明示或暗示的保证或条件。
# 有关许可证的特定权限和限制，请参见许可证。

# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入可选类型、元组和联合类型
from typing import Optional, Tuple, Union

# 导入 flax 库
import flax
# 导入 jax 的 numpy 模块
import jax.numpy as jnp

# 从配置工具中导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从调度工具中导入相关类和函数
from .scheduling_utils_flax import (
    CommonSchedulerState,  # 导入通用调度器状态
    FlaxKarrasDiffusionSchedulers,  # 导入 Karras 扩散调度器
    FlaxSchedulerMixin,  # 导入调度器混合类
    FlaxSchedulerOutput,  # 导入调度器输出类
    broadcast_to_shape_from_left,  # 导入从左侧广播到形状的函数
)

# 定义一个调度器状态的数据类，使用 flax 的结构化数据类装饰器
@flax.struct.dataclass
class EulerDiscreteSchedulerState:
    common: CommonSchedulerState  # 包含通用调度器状态的属性

    # 可设置的值
    init_noise_sigma: jnp.ndarray  # 初始噪声的标准差
    timesteps: jnp.ndarray  # 时间步长的数组
    sigmas: jnp.ndarray  # sigma 值的数组
    num_inference_steps: Optional[int] = None  # 推理步骤的数量，可选

    # 类方法，用于创建 EulerDiscreteSchedulerState 实例
    @classmethod
    def create(
        cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray, sigmas: jnp.ndarray
    ):
        # 返回一个新的实例，使用提供的参数初始化
        return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=timesteps, sigmas=sigmas)

# 定义一个输出类，继承自 FlaxSchedulerOutput
@dataclass
class FlaxEulerDiscreteSchedulerOutput(FlaxSchedulerOutput):
    state: EulerDiscreteSchedulerState  # 包含调度器状态的属性

# 定义一个调度器类，继承自 FlaxSchedulerMixin 和 ConfigMixin
class FlaxEulerDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    Euler 调度器（算法 2），参考 Karras 等人（2022） https://arxiv.org/abs/2206.00364。基于 Katherine Crowson 的原始
    k-diffusion 实现：
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L51

    [`~ConfigMixin`] 处理在调度器的 `__init__` 函数中传递的所有配置属性的存储，
    例如 `num_train_timesteps`。可以通过 `scheduler.config.num_train_timesteps` 访问它们。
    [`SchedulerMixin`] 提供通过 [`SchedulerMixin.save_pretrained`] 和
    [`~SchedulerMixin.from_pretrained`] 函数进行通用加载和保存的功能。
    """
    # 文档字符串，描述类初始化方法的参数
    Args:
        num_train_timesteps (`int`): 模型训练所用的扩散步骤数量
        beta_start (`float`): 推理开始时的 `beta` 值
        beta_end (`float`): 最终的 `beta` 值
        beta_schedule (`str`):
            beta 调度，表示从 beta 范围到模型步进的 beta 序列的映射。可选值为
            `linear` 或 `scaled_linear`
        trained_betas (`jnp.ndarray`, optional):
            直接传递 beta 数组给构造函数以绕过 `beta_start`、`beta_end` 等选项
        prediction_type (`str`, default `epsilon`, optional):
            调度函数的预测类型，选项包括 `epsilon`（预测扩散过程的噪声）、`sample`（直接预测噪声样本）或 `v_prediction`（见第 2.4 节 https://imagen.research.google/video/paper.pdf）
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            用于参数和计算的 `dtype`
    """

    # 获取所有兼容的 FlaxKarrasDiffusionSchedulers 名称
    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    # 声明 dtype 变量，类型为 jnp.dtype
    dtype: jnp.dtype

    # 属性，指示该类是否有状态
    @property
    def has_state(self):
        # 返回 True，表示该调度器具有状态
        return True

    # 注册到配置的初始化方法
    @register_to_config
    def __init__(
        # 设置训练步骤数量，默认为 1000
        num_train_timesteps: int = 1000,
        # 设置推理起始 beta 值，默认为 0.0001
        beta_start: float = 0.0001,
        # 设置推理结束 beta 值，默认为 0.02
        beta_end: float = 0.02,
        # 设置 beta 调度，默认为 "linear"
        beta_schedule: str = "linear",
        # 可选参数，直接传递 beta 数组
        trained_betas: Optional[jnp.ndarray] = None,
        # 设置预测类型，默认为 "epsilon"
        prediction_type: str = "epsilon",
        # 设置时间步间距，默认为 "linspace"
        timestep_spacing: str = "linspace",
        # 设置数据类型，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32,
    ):
        # 将 dtype 赋值给实例变量
        self.dtype = dtype

    # 创建状态的方法，接受可选的公共状态参数
    def create_state(self, common: Optional[CommonSchedulerState] = None) -> EulerDiscreteSchedulerState:
        # 如果没有传入公共状态，则创建一个新的
        if common is None:
            common = CommonSchedulerState.create(self)

        # 生成时间步的数组，逆序排列
        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
        # 计算每个时间步的标准差
        sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5
        # 通过插值调整 sigmas 的值
        sigmas = jnp.interp(timesteps, jnp.arange(0, len(sigmas)), sigmas)
        # 在 sigmas 后附加一个 0.0 值
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])

        # 标准化初始噪声分布的标准差
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            # 若时间步间距为 linspace 或 trailing，取最大 sigmas 值
            init_noise_sigma = sigmas.max()
        else:
            # 否则计算初始化噪声标准差
            init_noise_sigma = (sigmas.max() ** 2 + 1) ** 0.5

        # 返回 EulerDiscreteSchedulerState 的实例，包含公共状态、初始噪声标准差、时间步和 sigmas
        return EulerDiscreteSchedulerState.create(
            common=common,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            sigmas=sigmas,
        )
    # 定义一个方法，缩放去噪模型输入，以匹配欧拉算法
    def scale_model_input(self, state: EulerDiscreteSchedulerState, sample: jnp.ndarray, timestep: int) -> jnp.ndarray:
        """
        缩放去噪模型输入，计算方式为 `(sigma**2 + 1) ** 0.5`，以匹配欧拉算法。

        参数:
            state (`EulerDiscreteSchedulerState`):
                `FlaxEulerDiscreteScheduler` 状态数据类实例。
            sample (`jnp.ndarray`):
                当前正在通过扩散过程创建的样本实例。
            timestep (`int`):
                扩散链中的当前离散时间步。

        返回:
            `jnp.ndarray`: 缩放后的输入样本
        """
        # 获取当前时间步对应的索引
        (step_index,) = jnp.where(state.timesteps == timestep, size=1)
        # 提取索引的第一个元素
        step_index = step_index[0]

        # 根据索引获取 sigma 值
        sigma = state.sigmas[step_index]
        # 将样本数据缩放
        sample = sample / ((sigma**2 + 1) ** 0.5)
        # 返回缩放后的样本
        return sample

    # 定义一个方法，设置扩散链中使用的时间步
    def set_timesteps(
        self, state: EulerDiscreteSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> EulerDiscreteSchedulerState:
        """
        设置扩散链中使用的时间步。支持在推理之前运行的功能。

        参数:
            state (`EulerDiscreteSchedulerState`):
                `FlaxEulerDiscreteScheduler` 状态数据类实例。
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数。
        """

        # 根据配置的时间步间隔类型生成时间步
        if self.config.timestep_spacing == "linspace":
            # 生成线性间隔的时间步
            timesteps = jnp.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps, dtype=self.dtype)
        elif self.config.timestep_spacing == "leading":
            # 计算步骤比率，并生成时间步
            step_ratio = self.config.num_train_timesteps // num_inference_steps
            timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(float)
            timesteps += 1
        else:
            # 抛出异常，时间步间隔类型无效
            raise ValueError(
                f"timestep_spacing must be one of ['linspace', 'leading'], got {self.config.timestep_spacing}"
            )

        # 计算 sigma 值
        sigmas = ((1 - state.common.alphas_cumprod) / state.common.alphas_cumprod) ** 0.5
        # 在时间步上插值 sigma 值
        sigmas = jnp.interp(timesteps, jnp.arange(0, len(sigmas)), sigmas)
        # 在 sigma 后面附加 0.0
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])

        # 标准差初始化噪声分布
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            # 对于线性或尾随间隔，初始化噪声的 sigma 为 sigma 的最大值
            init_noise_sigma = sigmas.max()
        else:
            # 否则，计算初始化噪声的 sigma
            init_noise_sigma = (sigmas.max() ** 2 + 1) ** 0.5

        # 替换状态中的时间步和 sigma 信息
        return state.replace(
            timesteps=timesteps,
            sigmas=sigmas,
            num_inference_steps=num_inference_steps,
            init_noise_sigma=init_noise_sigma,
        )

    # 定义一个步骤方法，处理模型输出和当前状态
    def step(
        self,
        state: EulerDiscreteSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        return_dict: bool = True,
    ) -> Union[FlaxEulerDiscreteSchedulerOutput, Tuple]:
        """
        通过逆向 SDE 预测上一个时间步的样本。核心函数用于从学习到的模型输出（通常是预测的噪声）传播扩散过程。

        Args:
            state (`EulerDiscreteSchedulerState`):
                `FlaxEulerDiscreteScheduler` 状态数据类实例。
            model_output (`jnp.ndarray`): 来自学习扩散模型的直接输出。
            timestep (`int`): 当前扩散链中的离散时间步。
            sample (`jnp.ndarray`):
                当前通过扩散过程生成的样本实例。
            order: 多步推理的系数。
            return_dict (`bool`): 返回元组而非 FlaxEulerDiscreteScheduler 类的选项。

        Returns:
            [`FlaxEulerDiscreteScheduler`] 或 `tuple`: 如果 `return_dict` 为 True，则返回 [`FlaxEulerDiscreteScheduler`]，
            否则返回一个 `tuple`。返回元组时，第一个元素是样本张量。

        """
        # 检查推理步骤数量是否为 None，如果是，则抛出错误
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 获取当前时间步在 timesteps 中的索引
        (step_index,) = jnp.where(state.timesteps == timestep, size=1)
        step_index = step_index[0]  # 提取索引值

        # 获取当前时间步对应的 sigma 值
        sigma = state.sigmas[step_index]

        # 1. 从 sigma 缩放的预测噪声计算预测的原始样本 (x_0)
        if self.config.prediction_type == "epsilon":
            # 使用 epsilon 进行预测
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # 使用 v_prediction 进行预测，计算方法涉及线性组合
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            # 如果提供的 prediction_type 无效，抛出错误
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. 转换为 ODE 导数
        derivative = (sample - pred_original_sample) / sigma

        # 计算 dt，表示当前 sigma 与下一个 sigma 的差值
        dt = state.sigmas[step_index + 1] - sigma

        # 计算前一个样本的值
        prev_sample = sample + derivative * dt

        # 如果不返回字典，则返回元组
        if not return_dict:
            return (prev_sample, state)

        # 否则返回 FlaxEulerDiscreteSchedulerOutput 对象
        return FlaxEulerDiscreteSchedulerOutput(prev_sample=prev_sample, state=state)

    def add_noise(
        self,
        state: EulerDiscreteSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        # 获取当前时间步对应的 sigma 值，并展平为一维
        sigma = state.sigmas[timesteps].flatten()
        # 将 sigma 广播到与噪声形状相同
        sigma = broadcast_to_shape_from_left(sigma, noise.shape)

        # 生成加噪声的样本
        noisy_samples = original_samples + noise * sigma

        # 返回加噪声的样本
        return noisy_samples

    def __len__(self):
        # 返回训练时间步的数量
        return self.config.num_train_timesteps
```