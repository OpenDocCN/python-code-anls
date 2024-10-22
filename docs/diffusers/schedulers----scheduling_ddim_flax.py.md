# `.\diffusers\schedulers\scheduling_ddim_flax.py`

```py
# 版权所有 2024 斯坦福大学团队与 HuggingFace 团队，保留所有权利。
#
# 根据 Apache 许可证，第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是按“现状”基础提供，
# 不附带任何明示或暗示的担保或条件。
# 有关许可证所治理权限和限制的具体语言，请参见许可证。

# 免责声明：此代码受到以下项目的强烈影响：https://github.com/pesser/pytorch_diffusion
# 和 https://github.com/hojonathanho/diffusion

# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入可选类型、元组和联合类型
from typing import Optional, Tuple, Union

# 导入 flax 库
import flax
# 导入 JAX 的 numpy 模块并命名为 jnp
import jax.numpy as jnp

# 从配置工具和调度工具导入所需组件
from ..configuration_utils import ConfigMixin, register_to_config
from .scheduling_utils_flax import (
    CommonSchedulerState,  # 导入通用调度器状态
    FlaxKarrasDiffusionSchedulers,  # 导入 Karras Diffusion 调度器
    FlaxSchedulerMixin,  # 导入调度器混合功能
    FlaxSchedulerOutput,  # 导入调度器输出类型
    add_noise_common,  # 导入添加噪声的公共函数
    get_velocity_common,  # 导入获取速度的公共函数
)

# 定义 DDIM 调度器状态的结构数据类
@flax.struct.dataclass
class DDIMSchedulerState:
    common: CommonSchedulerState  # 包含通用调度器状态
    final_alpha_cumprod: jnp.ndarray  # 最终的 alpha 累积产品

    # 可设置的值
    init_noise_sigma: jnp.ndarray  # 初始噪声标准差
    timesteps: jnp.ndarray  # 时间步数组
    num_inference_steps: Optional[int] = None  # 可选的推理步骤数

    # 类方法，用于创建 DDIMSchedulerState 实例
    @classmethod
    def create(
        cls,
        common: CommonSchedulerState,  # 传入的通用调度器状态
        final_alpha_cumprod: jnp.ndarray,  # 传入的最终 alpha 累积产品
        init_noise_sigma: jnp.ndarray,  # 传入的初始噪声标准差
        timesteps: jnp.ndarray,  # 传入的时间步数组
    ):
        # 返回新的 DDIMSchedulerState 实例
        return cls(
            common=common,  # 设置通用调度器状态
            final_alpha_cumprod=final_alpha_cumprod,  # 设置最终 alpha 累积产品
            init_noise_sigma=init_noise_sigma,  # 设置初始噪声标准差
            timesteps=timesteps,  # 设置时间步数组
        )

# 定义 FlaxDDIMSchedulerOutput 数据类，继承自 FlaxSchedulerOutput
@dataclass
class FlaxDDIMSchedulerOutput(FlaxSchedulerOutput):
    state: DDIMSchedulerState  # 状态为 DDIMSchedulerState 类型

# 定义 FlaxDDIMScheduler 类，继承自 FlaxSchedulerMixin 和 ConfigMixin
class FlaxDDIMScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    去噪扩散隐式模型是一种调度器，它扩展了在去噪扩散概率模型 (DDPMs) 中引入的去噪过程
    ，并具有非马尔可夫引导。

    [`~ConfigMixin`] 负责存储在调度器的 `__init__` 函数中传递的所有配置属性，
    例如 `num_train_timesteps`。它们可以通过 `scheduler.config.num_train_timesteps` 访问。
    [`SchedulerMixin`] 通过 [`SchedulerMixin.save_pretrained`] 和
    [`~SchedulerMixin.from_pretrained`] 函数提供一般的加载和保存功能。

    有关更多详细信息，请参见原始论文：https://arxiv.org/abs/2010.02502
    # 函数参数说明
    Args:
        # 训练模型使用的扩散步骤数量
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        # 推理过程开始的 beta 值
        beta_start (`float`): the starting `beta` value of inference.
        # 推理过程最终的 beta 值
        beta_end (`float`): the final `beta` value.
        # beta 调度方式
        beta_schedule (`str`):
            # beta 范围到 beta 序列的映射，选择步进模型的方式，可选值包括
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        # 直接传递 beta 数组的选项，以绕过 beta_start、beta_end 等参数
        trained_betas (`jnp.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        # 预测样本的裁剪选项，用于数值稳定性
        clip_sample (`bool`, default `True`):
            option to clip predicted sample between for numerical stability. The clip range is determined by
            `clip_sample_range`.
        # 裁剪的最大幅度，仅在 clip_sample=True 时有效
        clip_sample_range (`float`, default `1.0`):
            the maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        # 将每个扩散步骤的前一个 alpha 乘积固定为 1 的选项
        set_alpha_to_one (`bool`, default `True`):
            each diffusion step uses the value of alphas product at that step and at the previous one. For the final
            step there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the value of alpha at step 0.
        # 推理步骤的偏移量
        steps_offset (`int`, default `0`):
            An offset added to the inference steps, as required by some model families.
        # 指示模型预测噪声或样本的类型
        prediction_type (`str`, default `epsilon`):
            indicates whether the model predicts the noise (epsilon), or the samples. One of `epsilon`, `sample`.
            `v-prediction` is not supported for this scheduler.
        # 用于参数和计算的数据类型
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            the `dtype` used for params and computation.
    """

    # 创建一个兼容的调度器列表，包含 FlaxKarrasDiffusionSchedulers 中的调度器名称
    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    # 定义数据类型
    dtype: jnp.dtype

    # 定义一个属性，表示该类是否有状态
    @property
    def has_state(self):
        # 返回 True，表示该类有状态
        return True

    # 注册到配置中的构造函数
    @register_to_config
    def __init__(
        # 训练步骤数量，默认为 1000
        num_train_timesteps: int = 1000,
        # 初始 beta 值，默认为 0.0001
        beta_start: float = 0.0001,
        # 最终 beta 值，默认为 0.02
        beta_end: float = 0.02,
        # beta 调度方式，默认为 "linear"
        beta_schedule: str = "linear",
        # 训练 beta 的数组，默认为 None
        trained_betas: Optional[jnp.ndarray] = None,
        # 是否裁剪样本，默认为 True
        clip_sample: bool = True,
        # 样本裁剪范围，默认为 1.0
        clip_sample_range: float = 1.0,
        # 是否将 alpha 的前一个值设置为 1，默认为 True
        set_alpha_to_one: bool = True,
        # 步骤偏移量，默认为 0
        steps_offset: int = 0,
        # 预测类型，默认为 "epsilon"
        prediction_type: str = "epsilon",
        # 数据类型，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32,
    ):
        # 将输入的数据类型赋值给实例变量
        self.dtype = dtype
    # 创建状态的方法，接收一个可选的公共调度器状态参数，返回 DDIM 调度器状态
    def create_state(self, common: Optional[CommonSchedulerState] = None) -> DDIMSchedulerState:
        # 如果未提供公共调度器状态，则创建一个新的公共调度器状态
        if common is None:
            common = CommonSchedulerState.create(self)

        # 在每个 DDIM 步骤中，我们查看前一个 alphas_cumprod
        # 对于最后一步，没有前一个 alphas_cumprod，因为我们已经处于 0
        # `set_alpha_to_one` 决定我们是将该参数设置为 1，还是使用“非前一个”最终 alpha
        final_alpha_cumprod = (
            # 如果设置为 1，则使用 1.0 的数组，否则使用公共状态中的第一个 alphas_cumprod
            jnp.array(1.0, dtype=self.dtype) if self.config.set_alpha_to_one else common.alphas_cumprod[0]
        )

        # 初始化噪声分布的标准差
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)

        # 生成从 0 到训练时间步数的数组，并反转顺序
        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]

        # 创建并返回 DDIM 调度器状态
        return DDIMSchedulerState.create(
            common=common,
            final_alpha_cumprod=final_alpha_cumprod,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )

    # 规模模型输入的方法，接收状态、样本和可选的时间步
    def scale_model_input(
        self, state: DDIMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
    ) -> jnp.ndarray:
        """
        参数:
            state (`PNDMSchedulerState`): `FlaxPNDMScheduler` 状态数据类实例。
            sample (`jnp.ndarray`): 输入样本
            timestep (`int`, optional): 当前时间步

        返回:
            `jnp.ndarray`: 缩放后的输入样本
        """
        # 直接返回输入样本，不做任何处理
        return sample

    # 设置时间步的方法，接收状态、推理步骤数量和形状
    def set_timesteps(
        self, state: DDIMSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> DDIMSchedulerState:
        """
        设置用于扩散链的离散时间步。支持在推理之前运行的功能。

        参数:
            state (`DDIMSchedulerState`):
                `FlaxDDIMScheduler` 状态数据类实例。
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数量。
        """
        # 计算步骤比率，通过训练时间步数除以推理步骤数量
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        # 通过比率生成整数时间步，乘以比率
        # 四舍五入以避免当 num_inference_step 为 3 的幂时出现问题
        timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round()[::-1] + self.config.steps_offset

        # 用新的时间步数和推理步骤数量替换状态
        return state.replace(
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
        )
    # 计算在给定时间步的方差
    def _get_variance(self, state: DDIMSchedulerState, timestep, prev_timestep):
        # 获取当前时间步的累计alpha值
        alpha_prod_t = state.common.alphas_cumprod[timestep]
        # 根据前一个时间步获取其累计alpha值，如果为负则使用最终的累计alpha值
        alpha_prod_t_prev = jnp.where(
            prev_timestep >= 0, state.common.alphas_cumprod[prev_timestep], state.final_alpha_cumprod
        )
        # 计算当前时间步的beta值
        beta_prod_t = 1 - alpha_prod_t
        # 计算前一个时间步的beta值
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 根据当前和前一个时间步的beta和alpha值计算方差
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        # 返回计算得到的方差
        return variance

    # 执行单步更新
    def step(
        self,
        state: DDIMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        eta: float = 0.0,
        return_dict: bool = True,
    # 添加噪声到原始样本
    def add_noise(
        self,
        state: DDIMSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        # 调用通用函数添加噪声并返回结果
        return add_noise_common(state.common, original_samples, noise, timesteps)

    # 计算样本的速度
    def get_velocity(
        self,
        state: DDIMSchedulerState,
        sample: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        # 调用通用函数获取速度并返回结果
        return get_velocity_common(state.common, sample, noise, timesteps)

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```