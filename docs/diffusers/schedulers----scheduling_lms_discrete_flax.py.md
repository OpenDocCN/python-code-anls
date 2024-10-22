# `.\diffusers\schedulers\scheduling_lms_discrete_flax.py`

```py
# 版权声明，说明文件的版权归2024年Katherine Crowson及HuggingFace团队所有
# 授权信息，指出该文件受Apache许可证2.0版的保护
# 用户只能在遵守许可证的情况下使用该文件
# 用户可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面协议，软件在"现状"基础上分发
# 不提供任何形式的明示或暗示的担保或条件
# 查看许可证以获取特定权限和限制信息

# 从dataclass模块导入dataclass装饰器
from dataclasses import dataclass
# 从typing模块导入Optional、Tuple和Union类型
from typing import Optional, Tuple, Union

# 导入flax库
import flax
# 从jax.numpy导入jnp模块，用于数值计算
import jax.numpy as jnp
# 从scipy库导入integrate模块，用于数值积分
from scipy import integrate

# 从configuration_utils模块导入ConfigMixin和register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从scheduling_utils_flax模块导入相关的调度器类
from .scheduling_utils_flax import (
    CommonSchedulerState,  # 通用调度器状态
    FlaxKarrasDiffusionSchedulers,  # Karras扩散调度器
    FlaxSchedulerMixin,  # 调度器混合类
    FlaxSchedulerOutput,  # 调度器输出类
    broadcast_to_shape_from_left,  # 从左侧广播形状的函数
)

# 定义LMSDiscreteSchedulerState类，表示调度器的状态
@flax.struct.dataclass
class LMSDiscreteSchedulerState:
    common: CommonSchedulerState  # 通用调度器状态的实例

    # 可设置的属性
    init_noise_sigma: jnp.ndarray  # 初始化噪声标准差
    timesteps: jnp.ndarray  # 时间步数组
    sigmas: jnp.ndarray  # 噪声标准差数组
    num_inference_steps: Optional[int] = None  # 可选的推理步骤数量

    # 运行时的属性
    derivatives: Optional[jnp.ndarray] = None  # 可选的导数数组

    # 类方法，用于创建LMSDiscreteSchedulerState实例
    @classmethod
    def create(
        cls, common: CommonSchedulerState, init_noise_sigma: jnp.ndarray, timesteps: jnp.ndarray, sigmas: jnp.ndarray
    ):
        # 返回一个新的LMSDiscreteSchedulerState实例
        return cls(common=common, init_noise_sigma=init_noise_sigma, timesteps=timesteps, sigmas=sigmas)

# 定义FlaxLMSSchedulerOutput类，表示调度器的输出
@dataclass
class FlaxLMSSchedulerOutput(FlaxSchedulerOutput):
    state: LMSDiscreteSchedulerState  # LMSDiscreteSchedulerState的实例

# 定义FlaxLMSDiscreteScheduler类，表示线性多步调度器
class FlaxLMSDiscreteScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    线性多步调度器，用于离散beta调度。基于Katherine Crowson的原始k-diffusion实现：
    https://github.com/crowsonkb/k-diffusion/blob/481677d114f6ea445aa009cf5bd7a9cdee909e47/k_diffusion/sampling.py#L181

    [`~ConfigMixin`]负责存储传递给调度器`__init__`函数的所有配置属性，例如`num_train_timesteps`。
    可以通过`scheduler.config.num_train_timesteps`访问。
    [`SchedulerMixin`]提供通过[`SchedulerMixin.save_pretrained`]和
    [`~SchedulerMixin.from_pretrained`]函数进行的通用加载和保存功能。
    """
    # 参数说明
    Args:
        num_train_timesteps (`int`): 训练模型时使用的扩散步骤数。
        beta_start (`float`): 推理时的起始 `beta` 值。
        beta_end (`float`): 最终 `beta` 值。
        beta_schedule (`str`):
            beta 调度，表示从 beta 范围到一系列 beta 的映射，用于模型的步进。可选择
            `linear` 或 `scaled_linear`。
        trained_betas (`jnp.ndarray`, optional):
            直接传递 beta 数组到构造函数的选项，以绕过 `beta_start`、`beta_end` 等。
        prediction_type (`str`, default `epsilon`, optional):
            调度函数的预测类型，可能值有 `epsilon`（预测扩散过程的噪声）、`sample`（直接预测带噪声的样本）或 `v_prediction`（见第 2.4 节
            https://imagen.research.google/video/paper.pdf）。
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`):
            用于参数和计算的 `dtype` 类型。
    """

    # 创建一个包含 FlaxKarrasDiffusionSchedulers 中每个调度器名称的列表
    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    # 定义一个数据类型属性
    dtype: jnp.dtype

    # 定义属性，指示是否有状态
    @property
    def has_state(self):
        return True

    # 注册构造函数到配置
    @register_to_config
    def __init__(
        # 初始化时的参数，设定默认值
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[jnp.ndarray] = None,
        prediction_type: str = "epsilon",
        dtype: jnp.dtype = jnp.float32,
    ):
        # 将传入的数据类型参数赋值给实例变量
        self.dtype = dtype

    # 创建状态的方法，接受一个可选的公共调度器状态
    def create_state(self, common: Optional[CommonSchedulerState] = None) -> LMSDiscreteSchedulerState:
        # 如果没有传入公共状态，则创建一个新的公共状态
        if common is None:
            common = CommonSchedulerState.create(self)

        # 生成一个从 0 到 num_train_timesteps 的时间步数组，并反转
        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
        # 计算每个时间步的标准差，使用公式
        sigmas = ((1 - common.alphas_cumprod) / common.alphas_cumprod) ** 0.5

        # 初始噪声分布的标准差
        init_noise_sigma = sigmas.max()

        # 创建并返回一个 LMSDiscreteSchedulerState 实例，传入相关参数
        return LMSDiscreteSchedulerState.create(
            common=common,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
            sigmas=sigmas,
        )
    # 定义一个方法用于缩放模型输入以匹配 K-LMS 算法
    def scale_model_input(self, state: LMSDiscreteSchedulerState, sample: jnp.ndarray, timestep: int) -> jnp.ndarray:
        """
        通过 `(sigma**2 + 1) ** 0.5` 缩放去噪模型输入以匹配 K-LMS 算法。

        参数：
            state (`LMSDiscreteSchedulerState`):
                `FlaxLMSDiscreteScheduler` 状态数据类实例。
            sample (`jnp.ndarray`):
                当前由扩散过程创建的样本实例。
            timestep (`int`):
                扩散链中的当前离散时间步。

        返回：
            `jnp.ndarray`: 缩放后的输入样本
        """
        # 找到与当前时间步相等的索引
        (step_index,) = jnp.where(state.timesteps == timestep, size=1)
        # 获取索引的第一个值
        step_index = step_index[0]

        # 获取当前时间步对应的 sigma 值
        sigma = state.sigmas[step_index]
        # 将样本按缩放因子进行缩放
        sample = sample / ((sigma**2 + 1) ** 0.5)
        # 返回缩放后的样本
        return sample

    # 定义一个方法用于计算线性多步系数
    def get_lms_coefficient(self, state: LMSDiscreteSchedulerState, order, t, current_order):
        """
        计算线性多步系数。

        参数：
            order (TODO):
            t (TODO):
            current_order (TODO):
        """

        # 定义一个内部函数用于计算 LMS 导数
        def lms_derivative(tau):
            prod = 1.0
            # 遍历所有步长，计算导数的乘积
            for k in range(order):
                # 跳过当前的阶数
                if current_order == k:
                    continue
                # 计算导数乘积
                prod *= (tau - state.sigmas[t - k]) / (state.sigmas[t - current_order] - state.sigmas[t - k])
            # 返回导数值
            return prod

        # 使用数值积分计算集成系数
        integrated_coeff = integrate.quad(lms_derivative, state.sigmas[t], state.sigmas[t + 1], epsrel=1e-4)[0]

        # 返回集成系数
        return integrated_coeff

    # 定义一个方法用于设置扩散链使用的时间步
    def set_timesteps(
        self, state: LMSDiscreteSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> LMSDiscreteSchedulerState:
        """
        设置用于扩散链的时间步。在推理之前运行的辅助函数。

        参数：
            state (`LMSDiscreteSchedulerState`):
                `FlaxLMSDiscreteScheduler` 状态数据类实例。
            num_inference_steps (`int`):
                在生成样本时使用的扩散步骤数。
        """

        # 生成从最大训练时间步到 0 的线性时间步数组
        timesteps = jnp.linspace(self.config.num_train_timesteps - 1, 0, num_inference_steps, dtype=self.dtype)

        # 计算时间步的低索引和高索引
        low_idx = jnp.floor(timesteps).astype(jnp.int32)
        high_idx = jnp.ceil(timesteps).astype(jnp.int32)

        # 计算时间步的分数部分
        frac = jnp.mod(timesteps, 1.0)

        # 计算 sigma 值
        sigmas = ((1 - state.common.alphas_cumprod) / state.common.alphas_cumprod) ** 0.5
        # 插值计算 sigma 值
        sigmas = (1 - frac) * sigmas[low_idx] + frac * sigmas[high_idx]
        # 在 sigma 数组末尾添加 0.0
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0], dtype=self.dtype)])

        # 将时间步转换为整型
        timesteps = timesteps.astype(jnp.int32)

        # 初始化导数的值
        derivatives = jnp.zeros((0,) + shape, dtype=self.dtype)

        # 返回更新后的状态
        return state.replace(
            timesteps=timesteps,
            sigmas=sigmas,
            num_inference_steps=num_inference_steps,
            derivatives=derivatives,
        )
    # 定义一个方法，用于在扩散过程中预测上一个时间步的样本
    def step(
            self,
            state: LMSDiscreteSchedulerState,  # 当前调度器状态实例
            model_output: jnp.ndarray,  # 从学习到的扩散模型得到的直接输出
            timestep: int,  # 当前扩散链中的离散时间步
            sample: jnp.ndarray,  # 当前正在通过扩散过程生成的样本实例
            order: int = 4,  # 多步推理的系数
            return_dict: bool = True,  # 是否返回元组而非 FlaxLMSSchedulerOutput 类
    ) -> Union[FlaxLMSSchedulerOutput, Tuple]:
            """
            通过逆转 SDE 预测上一个时间步的样本。核心函数从学习到的模型输出（通常是预测噪声）传播扩散过程。
    
            Args:
                state (`LMSDiscreteSchedulerState`): FlaxLMSDiscreteScheduler 的状态数据类实例。
                model_output (`jnp.ndarray`): 从学习到的扩散模型直接输出。
                timestep (`int`): 当前离散时间步。
                sample (`jnp.ndarray`):
                    当前通过扩散过程创建的样本实例。
                order: 多步推理的系数。
                return_dict (`bool`): 是否返回元组而非 FlaxLMSSchedulerOutput 类。
    
            Returns:
                [`FlaxLMSSchedulerOutput`] or `tuple`: 如果 `return_dict` 为 True，返回 [`FlaxLMSSchedulerOutput`]，否则返回一个元组。当返回元组时，第一个元素是样本张量。
            """
            # 检查推理步骤是否为 None，如果是则抛出错误
            if state.num_inference_steps is None:
                raise ValueError(
                    "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
                )
    
            # 获取当前时间步的 sigma 值
            sigma = state.sigmas[timestep]
    
            # 1. 从 sigma 缩放的预测噪声计算预测的原始样本 (x_0)
            if self.config.prediction_type == "epsilon":
                # 计算预测的原始样本
                pred_original_sample = sample - sigma * model_output
            elif self.config.prediction_type == "v_prediction":
                # 使用 v 预测公式计算预测的原始样本
                pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
            else:
                # 如果 prediction_type 不符合预期，抛出错误
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
                )
    
            # 2. 转换为 ODE 导数
            derivative = (sample - pred_original_sample) / sigma  # 计算导数
            # 将新的导数添加到状态中
            state = state.replace(derivatives=jnp.append(state.derivatives, derivative))
            # 如果导数长度超过了设定的 order，删除最早的导数
            if len(state.derivatives) > order:
                state = state.replace(derivatives=jnp.delete(state.derivatives, 0))
    
            # 3. 计算线性多步系数
            order = min(timestep + 1, order)  # 确保 order 不超过当前时间步
            # 生成多步系数
            lms_coeffs = [self.get_lms_coefficient(state, order, timestep, curr_order) for curr_order in range(order)]
    
            # 4. 基于导数路径计算上一个样本
            prev_sample = sample + sum(
                coeff * derivative for coeff, derivative in zip(lms_coeffs, reversed(state.derivatives))
            )  # 计算上一个样本
    
            # 如果不需要返回字典，返回元组
            if not return_dict:
                return (prev_sample, state)
    
            # 返回 FlaxLMSSchedulerOutput 类实例
            return FlaxLMSSchedulerOutput(prev_sample=prev_sample, state=state)
    # 定义添加噪声的函数，接受调度状态、原始样本、噪声和时间步
    def add_noise(
            self,
            state: LMSDiscreteSchedulerState,
            original_samples: jnp.ndarray,
            noise: jnp.ndarray,
            timesteps: jnp.ndarray,
        ) -> jnp.ndarray:
        # 从调度状态中获取指定时间步的 sigma 值，并扁平化
        sigma = state.sigmas[timesteps].flatten()
        # 将 sigma 的形状广播到噪声的形状
        sigma = broadcast_to_shape_from_left(sigma, noise.shape)
        
        # 将噪声与原始样本结合，生成带噪声的样本
        noisy_samples = original_samples + noise * sigma
    
        # 返回带噪声的样本
        return noisy_samples
    
    # 定义获取对象长度的方法
    def __len__(self):
        # 返回训练时间步的数量
        return self.config.num_train_timesteps
```