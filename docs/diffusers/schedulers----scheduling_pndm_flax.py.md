# `.\diffusers\schedulers\scheduling_pndm_flax.py`

```py
# 版权所有 2024 浙江大学团队与 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，
# 否则根据许可证分发的软件按“原样”提供，
# 不附带任何形式的保证或条件，无论是明示或暗示的。
# 请参阅许可证以获取有关权限和
# 限制的具体语言。

# 免责声明：此文件受到 https://github.com/ermongroup/ddim 的强烈影响

# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入可选类型、元组和联合类型
from typing import Optional, Tuple, Union

# 导入 flax 库
import flax
# 导入 jax 库
import jax
# 导入 jax.numpy 模块并重命名为 jnp
import jax.numpy as jnp

# 从配置工具中导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从调度工具中导入多个调度相关的类和函数
from .scheduling_utils_flax import (
    CommonSchedulerState,  # 导入通用调度器状态
    FlaxKarrasDiffusionSchedulers,  # 导入 Flax Karras 扩散调度器
    FlaxSchedulerMixin,  # 导入 Flax 调度混合器
    FlaxSchedulerOutput,  # 导入 Flax 调度输出类
    add_noise_common,  # 导入通用添加噪声函数
)

# 定义 PNDMSchedulerState 类，使用 flax 的数据类装饰器
@flax.struct.dataclass
class PNDMSchedulerState:
    common: CommonSchedulerState  # 公共调度状态
    final_alpha_cumprod: jnp.ndarray  # 最终 alpha 的累积乘积

    # 可设置的值
    init_noise_sigma: jnp.ndarray  # 初始噪声标准差
    timesteps: jnp.ndarray  # 时间步数组
    num_inference_steps: Optional[int] = None  # 可选的推理步骤数
    prk_timesteps: Optional[jnp.ndarray] = None  # 可选的 Runge-Kutta 时间步
    plms_timesteps: Optional[jnp.ndarray] = None  # 可选的 PLMS 时间步

    # 运行时值
    cur_model_output: Optional[jnp.ndarray] = None  # 当前模型输出
    counter: Optional[jnp.int32] = None  # 计数器
    cur_sample: Optional[jnp.ndarray] = None  # 当前样本
    ets: Optional[jnp.ndarray] = None  # 可选的扩散状态数组

    # 定义一个类方法，用于创建 PNDMSchedulerState 实例
    @classmethod
    def create(
        cls,  # 类本身
        common: CommonSchedulerState,  # 传入的公共调度状态
        final_alpha_cumprod: jnp.ndarray,  # 传入的最终 alpha 累积乘积
        init_noise_sigma: jnp.ndarray,  # 传入的初始噪声标准差
        timesteps: jnp.ndarray,  # 传入的时间步数组
    ):
        # 返回一个 PNDMSchedulerState 实例
        return cls(
            common=common,
            final_alpha_cumprod=final_alpha_cumprod,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )

# 定义 FlaxPNDMSchedulerOutput 类，继承 FlaxSchedulerOutput
@dataclass
class FlaxPNDMSchedulerOutput(FlaxSchedulerOutput):
    state: PNDMSchedulerState  # PNDMSchedulerState 状态

# 定义 FlaxPNDMScheduler 类，继承 FlaxSchedulerMixin 和 ConfigMixin
class FlaxPNDMScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    Pseudo numerical methods for diffusion models (PNDM) proposes using more advanced ODE integration techniques,
    namely Runge-Kutta method and a linear multi-step method.

    [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
    function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
    [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and
    [`~SchedulerMixin.from_pretrained`] functions.

    For more details, see the original paper: https://arxiv.org/abs/2202.09778
    """  # 类的文档字符串，描述了调度器的功能和来源
    # 参数说明
    Args:
        num_train_timesteps (`int`): 训练模型所使用的扩散步骤数量。
        beta_start (`float`): 推理的起始 `beta` 值。
        beta_end (`float`): 最终的 `beta` 值。
        beta_schedule (`str`): 
            beta 调度，表示从一个 beta 范围到一系列 beta 的映射，用于模型的步骤选择。可选值为
            `linear`、`scaled_linear` 或 `squaredcos_cap_v2`。
        trained_betas (`jnp.ndarray`, optional): 
            可选参数，直接将 beta 数组传递给构造函数，以跳过 `beta_start`、`beta_end` 等设置。
        skip_prk_steps (`bool`): 
            允许调度器跳过原论文中定义的 Runge-Kutta 步骤，这些步骤在 plms 步骤之前是必要的；默认为 `False`。
        set_alpha_to_one (`bool`, default `False`): 
            每个扩散步骤使用该步骤和前一个步骤的 alpha 乘积的值。对于最后一步没有前一个 alpha。当此选项为 `True` 时，前一个 alpha 乘积固定为 `1`，否则使用步骤 0 的 alpha 值。
        steps_offset (`int`, default `0`): 
            添加到推理步骤的偏移量，某些模型系列需要此偏移。
        prediction_type (`str`, default `epsilon`, optional): 
            调度函数的预测类型，选项包括 `epsilon`（预测扩散过程中的噪声）、`sample`（直接预测带噪声的样本）或 `v_prediction`（见文献 2.4 https://imagen.research.google/video/paper.pdf）。
        dtype (`jnp.dtype`, *optional*, defaults to `jnp.float32`): 
            用于参数和计算的 `dtype` 类型。
    """

    # 获取 FlaxKarrasDiffusionSchedulers 中所有兼容的名称
    _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]

    # 定义数据类型
    dtype: jnp.dtype
    # 定义 PNDM 的阶数
    pndm_order: int

    # 定义属性以检查是否具有状态
    @property
    def has_state(self):
        # 返回 True，表示该对象具有状态
        return True

    # 注册到配置中，并定义初始化函数
    @register_to_config
    def __init__(
        # 设置训练时的扩散步骤数量，默认为 1000
        num_train_timesteps: int = 1000,
        # 设置推理的起始 beta 值，默认为 0.0001
        beta_start: float = 0.0001,
        # 设置最终的 beta 值，默认为 0.02
        beta_end: float = 0.02,
        # 设置 beta 调度类型，默认为 "linear"
        beta_schedule: str = "linear",
        # 可选参数，传递已训练的 beta 数组
        trained_betas: Optional[jnp.ndarray] = None,
        # 设置是否跳过 Runge-Kutta 步骤，默认为 False
        skip_prk_steps: bool = False,
        # 设置是否将 alpha 固定为 1，默认为 False
        set_alpha_to_one: bool = False,
        # 设置推理步骤的偏移量，默认为 0
        steps_offset: int = 0,
        # 设置预测类型，默认为 "epsilon"
        prediction_type: str = "epsilon",
        # 设置数据类型，默认为 jnp.float32
        dtype: jnp.dtype = jnp.float32,
    ):
        # 将数据类型赋值给实例变量
        self.dtype = dtype

        # 当前仅支持 F-PNDM，即 Runge-Kutta 方法
        # 有关算法的更多信息，请参见论文：https://arxiv.org/pdf/2202.09778.pdf
        # 主要查看公式 (9)、(12)、(13) 和算法 2。
        # 将 PNDM 阶数设置为 4
        self.pndm_order = 4
    # 创建状态的方法，接受一个可选的 CommonSchedulerState 参数
    def create_state(self, common: Optional[CommonSchedulerState] = None) -> PNDMSchedulerState:
        # 如果 common 参数为 None，则创建一个新的 CommonSchedulerState 实例
        if common is None:
            common = CommonSchedulerState.create(self)

        # 在每个 ddim 步骤中，我们查看前一个 alphas_cumprod
        # 对于最后一步，由于我们已经处于 0，因此没有前一个 alphas_cumprod
        # `set_alpha_to_one` 决定我们是否将该参数简单设置为 1，还是
        # 使用“非前一个”的最终 alpha。
        final_alpha_cumprod = (
            jnp.array(1.0, dtype=self.dtype) if self.config.set_alpha_to_one else common.alphas_cumprod[0]
        )

        # 初始噪声分布的标准差
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)

        # 创建一个反向的时间步数组，从 num_train_timesteps 开始
        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]

        # 返回一个新的 PNDMSchedulerState 实例，包含 common、final_alpha_cumprod、init_noise_sigma 和 timesteps
        return PNDMSchedulerState.create(
            common=common,
            final_alpha_cumprod=final_alpha_cumprod,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )
    # 设置用于扩散链的离散时间步，推理前运行的辅助函数
    def set_timesteps(self, state: PNDMSchedulerState, num_inference_steps: int, shape: Tuple) -> PNDMSchedulerState:
        """
        设置用于扩散链的离散时间步，推理前运行的辅助函数。
    
        参数：
            state (`PNDMSchedulerState`):
                `FlaxPNDMScheduler` 状态数据类实例。
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数量。
            shape (`Tuple`):
                要生成的样本形状。
        """
    
        # 计算每个推理步骤的步长比
        step_ratio = self.config.num_train_timesteps // num_inference_steps
        # 通过乘以比率生成整数时间步
        # 四舍五入以避免 num_inference_step 为 3 的幂时出现问题
        _timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round() + self.config.steps_offset
    
        if self.config.skip_prk_steps:
            # 对于某些模型（如稳定扩散），可以/应该跳过 prk 步骤以产生更好的结果。
            # 使用 PNDM 时，如果配置跳过 prk 步骤，基于 crowsonkb 的 PLMS 采样实现
            prk_timesteps = jnp.array([], dtype=jnp.int32)
            # 生成 plms 时间步，将最后的时间步反转并添加到前面
            plms_timesteps = jnp.concatenate([_timesteps[:-1], _timesteps[-2:-1], _timesteps[-1:]])[::-1]
    
        else:
            # 生成 prk 时间步，重复并添加偏移
            prk_timesteps = _timesteps[-self.pndm_order :].repeat(2) + jnp.tile(
                jnp.array([0, self.config.num_train_timesteps // num_inference_steps // 2], dtype=jnp.int32),
                self.pndm_order,
            )
    
            # 反转并去掉边界的 prk 时间步
            prk_timesteps = (prk_timesteps[:-1].repeat(2)[1:-1])[::-1]
            # 反转 plms 时间步
            plms_timesteps = _timesteps[:-3][::-1]
    
        # 合并 prk 和 plms 时间步
        timesteps = jnp.concatenate([prk_timesteps, plms_timesteps])
    
        # 初始化运行值
        # 创建当前模型输出的零数组，形状为传入的 shape
        cur_model_output = jnp.zeros(shape, dtype=self.dtype)
        # 初始化计数器为 0
        counter = jnp.int32(0)
        # 创建当前样本的零数组，形状为传入的 shape
        cur_sample = jnp.zeros(shape, dtype=self.dtype)
        # 创建一个额外的数组，用于存储中间结果
        ets = jnp.zeros((4,) + shape, dtype=self.dtype)
    
        # 返回更新后的状态，包含新的时间步和运行值
        return state.replace(
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            prk_timesteps=prk_timesteps,
            plms_timesteps=plms_timesteps,
            cur_model_output=cur_model_output,
            counter=counter,
            cur_sample=cur_sample,
            ets=ets,
        )
    
    # 定义缩放模型输入的函数
    def scale_model_input(
        self, state: PNDMSchedulerState, sample: jnp.ndarray, timestep: Optional[int] = None
    ) -> jnp.ndarray:
        # 声明函数返回类型为 jnp.ndarray（JAX 的 ndarray）
        """
        # 函数文档字符串，说明该函数的用途
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            # 参数 state，类型为 PNDMSchedulerState，表示调度器的状态数据类实例
            sample (`jnp.ndarray`): input sample
            # 参数 sample，类型为 jnp.ndarray，表示输入样本
            timestep (`int`, optional): current timestep
            # 可选参数 timestep，类型为 int，表示当前时间步

        Returns:
            `jnp.ndarray`: scaled input sample
            # 返回类型为 jnp.ndarray，表示缩放后的输入样本
        """
        return sample
        # 返回输入样本，当前未进行任何处理

    def step(
        # 定义 step 方法
        self,
        state: PNDMSchedulerState,
        # 参数 state，类型为 PNDMSchedulerState，表示调度器的状态数据类实例
        model_output: jnp.ndarray,
        # 参数 model_output，类型为 jnp.ndarray，表示模型的输出
        timestep: int,
        # 参数 timestep，类型为 int，表示当前时间步
        sample: jnp.ndarray,
        # 参数 sample，类型为 jnp.ndarray，表示输入样本
        return_dict: bool = True,
        # 参数 return_dict，类型为 bool，默认为 True，表示是否返回字典格式的结果
    ) -> Union[FlaxPNDMSchedulerOutput, Tuple]:
        """
        预测在上一个时间步的样本，通过反转 SDE。核心功能是从学习的模型输出传播扩散过程
        （通常是预测的噪声）。

        此函数根据内部变量 `counter` 调用 `step_prk()` 或 `step_plms()`。

        Args:
            state (`PNDMSchedulerState`): `FlaxPNDMScheduler` 状态数据类实例。
            model_output (`jnp.ndarray`): 来自学习扩散模型的直接输出。
            timestep (`int`): 当前扩散链中的离散时间步。
            sample (`jnp.ndarray`):
                正在通过扩散过程创建的当前样本实例。
            return_dict (`bool`): 返回元组而不是 `FlaxPNDMSchedulerOutput` 类的选项。

        Returns:
            [`FlaxPNDMSchedulerOutput`] 或 `tuple`: 如果 `return_dict` 为 True，则返回 [`FlaxPNDMSchedulerOutput`]，
            否则返回 `tuple`。返回元组时，第一个元素是样本张量。

        """

        # 检查推理步骤数量是否为 None，抛出错误以提醒用户
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 如果配置跳过 PRK 步骤，调用 PLMS 步骤
        if self.config.skip_prk_steps:
            prev_sample, state = self.step_plms(state, model_output, timestep, sample)
        else:
            # 否则，首先执行 PRK 步骤
            prk_prev_sample, prk_state = self.step_prk(state, model_output, timestep, sample)
            # 然后执行 PLMS 步骤
            plms_prev_sample, plms_state = self.step_plms(state, model_output, timestep, sample)

            # 检查当前计数器是否小于 PRK 时间步的长度
            cond = state.counter < len(state.prk_timesteps)

            # 根据条件选择前一个样本
            prev_sample = jax.lax.select(cond, prk_prev_sample, plms_prev_sample)

            # 更新状态，选择相应的当前模型输出和其他状态变量
            state = state.replace(
                cur_model_output=jax.lax.select(cond, prk_state.cur_model_output, plms_state.cur_model_output),
                ets=jax.lax.select(cond, prk_state.ets, plms_state.ets),
                cur_sample=jax.lax.select(cond, prk_state.cur_sample, plms_state.cur_sample),
                counter=jax.lax.select(cond, prk_state.counter, plms_state.counter),
            )

        # 如果不返回字典，则返回前一个样本和状态的元组
        if not return_dict:
            return (prev_sample, state)

        # 否则返回 FlaxPNDMSchedulerOutput 对象
        return FlaxPNDMSchedulerOutput(prev_sample=prev_sample, state=state)

    # 定义 step_prk 方法，用于执行 PRK 步骤
    def step_prk(
        self,
        state: PNDMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
    ) -> Union[FlaxPNDMSchedulerOutput, Tuple]:
        """
        Step function propagating the sample with the Runge-Kutta method. RK takes 4 forward passes to approximate the
        solution to the differential equation.

        Args:
            state (`PNDMSchedulerState`): the `FlaxPNDMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxPNDMSchedulerOutput class

        Returns:
            [`FlaxPNDMSchedulerOutput`] or `tuple`: [`FlaxPNDMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """

        # 检查推理步骤数量是否为 None，如果是则抛出异常
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 根据当前计数器决定与上一步的差值，计算上一步的时间步
        diff_to_prev = jnp.where(
            state.counter % 2, 0, self.config.num_train_timesteps // state.num_inference_steps // 2
        )
        prev_timestep = timestep - diff_to_prev  # 计算前一个时间步
        timestep = state.prk_timesteps[state.counter // 4 * 4]  # 更新当前时间步

        # 选择当前模型输出，基于计数器的余数决定逻辑
        model_output = jax.lax.select(
            (state.counter % 4) != 3,
            model_output,  # 余数为 0, 1, 2
            state.cur_model_output + 1 / 6 * model_output,  # 余数为 3
        )

        # 更新状态，替换当前模型输出、ets 和当前样本
        state = state.replace(
            cur_model_output=jax.lax.select_n(
                state.counter % 4,
                state.cur_model_output + 1 / 6 * model_output,  # 余数为 0
                state.cur_model_output + 1 / 3 * model_output,  # 余数为 1
                state.cur_model_output + 1 / 3 * model_output,  # 余数为 2
                jnp.zeros_like(state.cur_model_output),  # 余数为 3
            ),
            ets=jax.lax.select(
                (state.counter % 4) == 0,
                state.ets.at[0:3].set(state.ets[1:4]).at[3].set(model_output),  # 余数为 0
                state.ets,  # 余数为 1, 2, 3
            ),
            cur_sample=jax.lax.select(
                (state.counter % 4) == 0,
                sample,  # 余数为 0
                state.cur_sample,  # 余数为 1, 2, 3
            ),
        )

        cur_sample = state.cur_sample  # 获取当前样本
        # 获取前一个样本，基于当前状态和模型输出
        prev_sample = self._get_prev_sample(state, cur_sample, timestep, prev_timestep, model_output)
        # 更新状态计数器
        state = state.replace(counter=state.counter + 1)

        # 返回前一个样本和更新后的状态
        return (prev_sample, state)

    # 定义 step_plms 函数，参数包括状态、模型输出、时间步和样本
    def step_plms(
        self,
        state: PNDMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
    # 计算前一个样本，使用 PNDM 算法中的公式 (9)
    def _get_prev_sample(self, state: PNDMSchedulerState, sample, timestep, prev_timestep, model_output):
        # 查看 PNDM 论文中的公式 (9) 
        # 此函数使用公式 (9) 计算 x_(t−δ)
        # 注意：需要将 x_t 加到方程的两边

        # 符号约定 (<变量名> -> <论文中的名称>
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)

        # 获取当前时间步的累积 α 值
        alpha_prod_t = state.common.alphas_cumprod[timestep]
        # 如果 prev_timestep 大于等于 0，获取前一个时间步的累积 α 值，否则使用最终的累积 α 值
        alpha_prod_t_prev = jnp.where(
            prev_timestep >= 0, state.common.alphas_cumprod[prev_timestep], state.final_alpha_cumprod
        )
        # 计算当前时间步的 β 值
        beta_prod_t = 1 - alpha_prod_t
        # 计算前一个时间步的 β 值
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 根据预测类型进行不同的处理
        if self.config.prediction_type == "v_prediction":
            # 使用公式调整模型输出
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        elif self.config.prediction_type != "epsilon":
            # 如果预测类型不符合要求，则抛出异常
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `v_prediction`"
            )

        # 计算样本系数，对应公式 (9) 中的分母部分加 1
        # 注意：公式简化后可得 (α_(t−δ) - α_t) / (sqrt(α_t) * (sqrt(α_(t−δ)) + sqr(α_t))) =
        # sqrt(α_(t−δ)) / sqrt(α_t)
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)

        # 计算模型输出的分母系数，对应公式 (9) 中 e_θ(x_t, t) 的分母
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)

        # 根据公式 (9) 计算前一个样本
        prev_sample = (
            sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        )

        # 返回计算得到的前一个样本
        return prev_sample

    # 添加噪声到样本中
    def add_noise(
        self,
        state: PNDMSchedulerState,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
    ) -> jnp.ndarray:
        # 调用公共函数添加噪声
        return add_noise_common(state.common, original_samples, noise, timesteps)

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```