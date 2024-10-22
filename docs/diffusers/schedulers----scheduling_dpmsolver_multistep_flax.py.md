# `.\diffusers\schedulers\scheduling_dpmsolver_multistep_flax.py`

```py
# 版权所有 2024 TSAIL 团队和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（"许可证"）许可；
# 除非遵循许可证，否则不得使用此文件。
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，按“原样”分发的软件不附带任何明示或暗示的担保或条件。
# 有关许可证所适用权限和限制的具体说明，请参见许可证。

# 免责声明：此文件受到 https://github.com/LuChengTHU/dpm-solver 的强烈影响

# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 导入类型相关的模块
from typing import List, Optional, Tuple, Union

# 导入 flax 库
import flax
# 导入 jax 库
import jax
# 导入 jax 数组处理模块
import jax.numpy as jnp

# 从配置工具导入配置混合类和注册函数
from ..configuration_utils import ConfigMixin, register_to_config
# 从调度工具导入常用的调度器状态和类
from .scheduling_utils_flax import (
    CommonSchedulerState,
    FlaxKarrasDiffusionSchedulers,
    FlaxSchedulerMixin,
    FlaxSchedulerOutput,
    add_noise_common,
)

# 定义 DPMSolverMultistepSchedulerState 数据类，表示调度器的状态
@flax.struct.dataclass
class DPMSolverMultistepSchedulerState:
    # 调度器的通用状态
    common: CommonSchedulerState
    # 当前时间步的 alpha 值
    alpha_t: jnp.ndarray
    # 当前时间步的 sigma 值
    sigma_t: jnp.ndarray
    # 当前时间步的 lambda 值
    lambda_t: jnp.ndarray

    # 可设置的值
    init_noise_sigma: jnp.ndarray  # 初始化噪声标准差
    timesteps: jnp.ndarray          # 时间步数组
    num_inference_steps: Optional[int] = None  # 推理步骤数（可选）

    # 运行时值
    model_outputs: Optional[jnp.ndarray] = None  # 模型输出（可选）
    lower_order_nums: Optional[jnp.int32] = None  # 较低阶数（可选）
    prev_timestep: Optional[jnp.int32] = None  # 上一个时间步（可选）
    cur_sample: Optional[jnp.ndarray] = None    # 当前样本（可选）

    # 定义类方法以创建调度器状态
    @classmethod
    def create(
        cls,
        common: CommonSchedulerState,
        alpha_t: jnp.ndarray,
        sigma_t: jnp.ndarray,
        lambda_t: jnp.ndarray,
        init_noise_sigma: jnp.ndarray,
        timesteps: jnp.ndarray,
    ):
        # 返回新的调度器状态实例
        return cls(
            common=common,
            alpha_t=alpha_t,
            sigma_t=sigma_t,
            lambda_t=lambda_t,
            init_noise_sigma=init_noise_sigma,
            timesteps=timesteps,
        )

# 定义 FlaxDPMSolverMultistepSchedulerOutput 数据类，表示调度器输出
@dataclass
class FlaxDPMSolverMultistepSchedulerOutput(FlaxSchedulerOutput):
    # 调度器的状态
    state: DPMSolverMultistepSchedulerState

# 定义 FlaxDPMSolverMultistepScheduler 类，继承调度器混合类和配置混合类
class FlaxDPMSolverMultistepScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    DPM-Solver（以及改进版 DPM-Solver++）是一个快速的专用高阶求解器，用于扩散 ODE，并提供收敛阶数保证。
    实证表明，使用 DPM-Solver 仅 20 步就能生成高质量样本，即使仅用 10 步也能生成相当不错的样本。

    有关更多详细信息，请参见原始论文： https://arxiv.org/abs/2206.00927 和 https://arxiv.org/abs/2211.01095

    目前，我们支持多步 DPM-Solver 适用于噪声预测模型和数据预测模型。
    我们建议使用 `solver_order=2` 进行引导采样，使用 `solver_order=3` 进行无条件采样。
    # 支持 Imagen 中的“动态阈值”方法，参考文献：https://arxiv.org/abs/2205.11487
        We also support the "dynamic thresholding" method in Imagen (https://arxiv.org/abs/2205.11487). 
        # 对于像素空间扩散模型，可以同时设置 `algorithm_type="dpmsolver++"` 和 `thresholding=True` 来使用动态阈值
        For pixel-space diffusion models, you can set both `algorithm_type="dpmsolver++"` and `thresholding=True` to use the dynamic 
        # 注意，阈值方法不适合于潜空间扩散模型（如 stable-diffusion）
        thresholding. Note that the thresholding method is unsuitable for latent-space diffusion models (such as 
        stable-diffusion).
    
        # `ConfigMixin` 负责存储在调度器的 `__init__` 函数中传递的所有配置属性，例如 `num_train_timesteps`
        [`~ConfigMixin`] takes care of storing all config attributes that are passed in the scheduler's `__init__`
        # 这些属性可以通过 `scheduler.config.num_train_timesteps` 访问
        function, such as `num_train_timesteps`. They can be accessed via `scheduler.config.num_train_timesteps`.
        # `SchedulerMixin` 提供通用的加载和保存功能，通过 [`SchedulerMixin.save_pretrained`] 和 [`~SchedulerMixin.from_pretrained`] 函数
        [`SchedulerMixin`] provides general loading and saving functionality via the [`SchedulerMixin.save_pretrained`] and 
        [`~SchedulerMixin.from_pretrained`] functions.
    
        # 有关更多详细信息，请参见原始论文： https://arxiv.org/abs/2206.00927 和 https://arxiv.org/abs/2211.01095
        For more details, see the original paper: https://arxiv.org/abs/2206.00927 and https://arxiv.org/abs/2211.01095
    
        # 兼容的调度器列表，从 FlaxKarrasDiffusionSchedulers 中提取名称
        _compatibles = [e.name for e in FlaxKarrasDiffusionSchedulers]
    
        # 数据类型变量
        dtype: jnp.dtype
    
        # 属性，返回是否有状态
        @property
        def has_state(self):
            return True
    
        # 注册到配置的初始化函数，定义多个参数的默认值
        @register_to_config
        def __init__(
            # 训练时间步数，默认为 1000
            num_train_timesteps: int = 1000,
            # beta 的起始值，默认为 0.0001
            beta_start: float = 0.0001,
            # beta 的结束值，默认为 0.02
            beta_end: float = 0.02,
            # beta 的调度方式，默认为 "linear"
            beta_schedule: str = "linear",
            # 已训练的 beta 值，可选
            trained_betas: Optional[jnp.ndarray] = None,
            # 解算器阶数，默认为 2
            solver_order: int = 2,
            # 预测类型，默认为 "epsilon"
            prediction_type: str = "epsilon",
            # 是否启用阈值处理，默认为 False
            thresholding: bool = False,
            # 动态阈值比例，默认为 0.995
            dynamic_thresholding_ratio: float = 0.995,
            # 采样最大值，默认为 1.0
            sample_max_value: float = 1.0,
            # 算法类型，默认为 "dpmsolver++"
            algorithm_type: str = "dpmsolver++",
            # 解算器类型，默认为 "midpoint"
            solver_type: str = "midpoint",
            # 最后阶段是否降低阶数，默认为 True
            lower_order_final: bool = True,
            # 时间步的间隔类型，默认为 "linspace"
            timestep_spacing: str = "linspace",
            # 数据类型，默认为 jnp.float32
            dtype: jnp.dtype = jnp.float32,
        ):
            # 将数据类型赋值给实例变量
            self.dtype = dtype
    # 创建状态的方法，接受一个可选的公共调度状态参数，返回 DPM 求解器多步调度状态
    def create_state(self, common: Optional[CommonSchedulerState] = None) -> DPMSolverMultistepSchedulerState:
        # 如果没有提供公共调度状态，则创建一个新的实例
        if common is None:
            common = CommonSchedulerState.create(self)
    
        # 当前仅支持 VP 类型的噪声调度
        alpha_t = jnp.sqrt(common.alphas_cumprod)  # 计算累积 alpha 的平方根
        sigma_t = jnp.sqrt(1 - common.alphas_cumprod)  # 计算 1 减去累积 alpha 的平方根
        lambda_t = jnp.log(alpha_t) - jnp.log(sigma_t)  # 计算 alpha_t 和 sigma_t 的对数差
    
        # DPM 求解器的设置
        if self.config.algorithm_type not in ["dpmsolver", "dpmsolver++"]:
            # 如果算法类型不在支持的列表中，则抛出未实现异常
            raise NotImplementedError(f"{self.config.algorithm_type} is not implemented for {self.__class__}")
        if self.config.solver_type not in ["midpoint", "heun"]:
            # 如果求解器类型不在支持的列表中，则抛出未实现异常
            raise NotImplementedError(f"{self.config.solver_type} is not implemented for {self.__class__}")
    
        # 初始化噪声分布的标准差
        init_noise_sigma = jnp.array(1.0, dtype=self.dtype)  # 创建一个值为 1.0 的数组，类型为实例的 dtype
    
        # 生成时间步的数组，从 0 到 num_train_timesteps，取整后反转
        timesteps = jnp.arange(0, self.config.num_train_timesteps).round()[::-1]
    
        # 创建并返回 DPM 求解器多步调度状态
        return DPMSolverMultistepSchedulerState.create(
            common=common,  # 传入公共调度状态
            alpha_t=alpha_t,  # 传入计算得到的 alpha_t
            sigma_t=sigma_t,  # 传入计算得到的 sigma_t
            lambda_t=lambda_t,  # 传入计算得到的 lambda_t
            init_noise_sigma=init_noise_sigma,  # 传入初始化噪声的标准差
            timesteps=timesteps,  # 传入时间步数组
        )
    
    # 设置时间步的方法，接受当前状态、推理步骤数和形状作为参数
    def set_timesteps(
        self, state: DPMSolverMultistepSchedulerState, num_inference_steps: int, shape: Tuple
    ) -> DPMSolverMultistepSchedulerState:  # 定义返回类型为 DPMSolverMultistepSchedulerState
        """  # 文档字符串，描述该函数的功能
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.  # 设置用于扩散链的离散时间步，支持在推断前运行的函数

        Args:  # 参数说明
            state (`DPMSolverMultistepSchedulerState`):  # 状态参数，类型为 DPMSolverMultistepSchedulerState
                the `FlaxDPMSolverMultistepScheduler` state data class instance.  # FlaxDPMSolverMultistepScheduler 的状态数据类实例
            num_inference_steps (`int`):  # 推断步骤数量参数，类型为 int
                the number of diffusion steps used when generating samples with a pre-trained model.  # 生成样本时使用的扩散步骤数量
            shape (`Tuple`):  # 样本形状参数，类型为元组
                the shape of the samples to be generated.  # 要生成的样本的形状
        """  # 文档字符串结束
        last_timestep = self.config.num_train_timesteps  # 获取训练时的最后时间步
        if self.config.timestep_spacing == "linspace":  # 检查时间步间距配置是否为线性空间
            timesteps = (  # 生成线性空间的时间步
                jnp.linspace(0, last_timestep - 1, num_inference_steps + 1)  # 生成从0到最后时间步的线性间隔
                .round()[::-1][:-1]  # 取反并去掉最后一个元素
                .astype(jnp.int32)  # 转换为整型
            )
        elif self.config.timestep_spacing == "leading":  # 检查时间步间距配置是否为前导
            step_ratio = last_timestep // (num_inference_steps + 1)  # 计算步骤比率
            # creates integer timesteps by multiplying by ratio  # 通过乘以比率创建整数时间步
            # casting to int to avoid issues when num_inference_step is power of 3  # 强制转换为整数以避免在 num_inference_step 为 3 的幂时的问题
            timesteps = (  # 生成前导时间步
                (jnp.arange(0, num_inference_steps + 1) * step_ratio)  # 创建范围并乘以步骤比率
                .round()[::-1][:-1]  # 取反并去掉最后一个元素
                .copy().astype(jnp.int32)  # 复制并转换为整型
            )
            timesteps += self.config.steps_offset  # 加上步骤偏移量
        elif self.config.timestep_spacing == "trailing":  # 检查时间步间距配置是否为后置
            step_ratio = self.config.num_train_timesteps / num_inference_steps  # 计算步骤比率
            # creates integer timesteps by multiplying by ratio  # 通过乘以比率创建整数时间步
            # casting to int to avoid issues when num_inference_step is power of 3  # 强制转换为整数以避免在 num_inference_step 为 3 的幂时的问题
            timesteps = jnp.arange(last_timestep, 0, -step_ratio)  # 从最后时间步到0生成时间步
            .round().copy().astype(jnp.int32)  # 四舍五入、复制并转换为整型
            timesteps -= 1  # 时间步减去1
        else:  # 如果没有匹配的时间步间距配置
            raise ValueError(  # 抛出值错误
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."  # 提示用户选择有效的时间步间距
            )

        # initial running values  # 初始化运行值

        model_outputs = jnp.zeros((self.config.solver_order,) + shape, dtype=self.dtype)  # 创建模型输出数组，初始化为零
        lower_order_nums = jnp.int32(0)  # 初始化低阶数字为0
        prev_timestep = jnp.int32(-1)  # 初始化前一个时间步为-1
        cur_sample = jnp.zeros(shape, dtype=self.dtype)  # 创建当前样本数组，初始化为零

        return state.replace(  # 返回更新后的状态
            num_inference_steps=num_inference_steps,  # 更新推断步骤数量
            timesteps=timesteps,  # 更新时间步
            model_outputs=model_outputs,  # 更新模型输出
            lower_order_nums=lower_order_nums,  # 更新低阶数字
            prev_timestep=prev_timestep,  # 更新前一个时间步
            cur_sample=cur_sample,  # 更新当前样本
        )

    def convert_model_output(  # 定义转换模型输出的函数
        self,  # 实例对象
        state: DPMSolverMultistepSchedulerState,  # 状态参数，类型为 DPMSolverMultistepSchedulerState
        model_output: jnp.ndarray,  # 模型输出参数，类型为 jnp.ndarray
        timestep: int,  # 当前时间步参数，类型为 int
        sample: jnp.ndarray,  # 样本参数，类型为 jnp.ndarray
    def dpm_solver_first_order_update(  # 定义一阶更新的扩散模型求解器函数
        self,  # 实例对象
        state: DPMSolverMultistepSchedulerState,  # 状态参数，类型为 DPMSolverMultistepSchedulerState
        model_output: jnp.ndarray,  # 模型输出参数，类型为 jnp.ndarray
        timestep: int,  # 当前时间步参数，类型为 int
        prev_timestep: int,  # 前一个时间步参数，类型为 int
        sample: jnp.ndarray,  # 样本参数，类型为 jnp.ndarray
    # 函数返回一个一阶DPM求解器的步骤结果，等效于DDIM
    ) -> jnp.ndarray:
        # 文档字符串，说明函数的用途及详细推导链接
        """
        One step for the first-order DPM-Solver (equivalent to DDIM).
        See https://arxiv.org/abs/2206.00927 for the detailed derivation.
    
        Args:
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            prev_timestep (`int`): previous discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
    
        Returns:
            `jnp.ndarray`: the sample tensor at the previous timestep.
        """
        # 将前一个时间步和当前时间步赋值给变量
        t, s0 = prev_timestep, timestep
        # 获取模型输出
        m0 = model_output
        # 获取当前和前一个时间步的lambda值
        lambda_t, lambda_s = state.lambda_t[t], state.lambda_t[s0]
        # 获取当前和前一个时间步的alpha值
        alpha_t, alpha_s = state.alpha_t[t], state.alpha_t[s0]
        # 获取当前和前一个时间步的sigma值
        sigma_t, sigma_s = state.sigma_t[t], state.sigma_t[s0]
        # 计算h值，表示lambda_t与lambda_s的差异
        h = lambda_t - lambda_s
        # 根据配置的算法类型选择相应的计算公式
        if self.config.algorithm_type == "dpmsolver++":
            # 计算当前样本的更新值，使用dpmsolver++公式
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (jnp.exp(-h) - 1.0)) * m0
        elif self.config.algorithm_type == "dpmsolver":
            # 计算当前样本的更新值，使用dpmsolver公式
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (jnp.exp(h) - 1.0)) * m0
        # 返回更新后的样本
        return x_t
    
    # 定义一个多步骤DPM求解器的二阶更新函数
    def multistep_dpm_solver_second_order_update(
        # 接受当前状态作为参数
        self,
        state: DPMSolverMultistepSchedulerState,
        # 接受模型输出列表作为参数
        model_output_list: jnp.ndarray,
        # 接受时间步列表作为参数
        timestep_list: List[int],
        # 接受前一个时间步作为参数
        prev_timestep: int,
        # 接受当前样本作为参数
        sample: jnp.ndarray,
    # 返回上一个时间步的样本张量
    ) -> jnp.ndarray:
        # DPM-Solver的二阶多步一步
    
        # 参数说明：
        # model_output_list：当前和后续时间步的扩散模型直接输出的列表
        # timestep：当前和后续离散时间步
        # prev_timestep：前一个离散时间步
        # sample：当前扩散过程中的样本实例
    
        # 返回值为上一个时间步的样本张量
        """
        # 从前一个时间步获取时间步
        t, s0, s1 = prev_timestep, timestep_list[-1], timestep_list[-2]
        # 从模型输出列表中获取当前和前一个时间步的输出
        m0, m1 = model_output_list[-1], model_output_list[-2]
        # 获取状态中当前和前两个时间步的lambda值
        lambda_t, lambda_s0, lambda_s1 = state.lambda_t[t], state.lambda_t[s0], state.lambda_t[s1]
        # 获取状态中当前和前两个时间步的alpha值
        alpha_t, alpha_s0 = state.alpha_t[t], state.alpha_t[s0]
        # 获取状态中当前和前一个时间步的sigma值
        sigma_t, sigma_s0 = state.sigma_t[t], state.sigma_t[s0]
        # 计算h和h_0的差值
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        # 计算r0值
        r0 = h_0 / h
        # D0和D1的值
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)
        # 根据算法类型判断处理方式
        if self.config.algorithm_type == "dpmsolver++":
            # 参考详细推导文献
            if self.config.solver_type == "midpoint":
                # 使用中点法计算x_t
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (jnp.exp(-h) - 1.0)) * D0
                    - 0.5 * (alpha_t * (jnp.exp(-h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                # 使用Heun法计算x_t
                x_t = (
                    (sigma_t / sigma_s0) * sample
                    - (alpha_t * (jnp.exp(-h) - 1.0)) * D0
                    + (alpha_t * ((jnp.exp(-h) - 1.0) / h + 1.0)) * D1
                )
        elif self.config.algorithm_type == "dpmsolver":
            # 参考详细推导文献
            if self.config.solver_type == "midpoint":
                # 使用中点法计算x_t
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (jnp.exp(h) - 1.0)) * D0
                    - 0.5 * (sigma_t * (jnp.exp(h) - 1.0)) * D1
                )
            elif self.config.solver_type == "heun":
                # 使用Heun法计算x_t
                x_t = (
                    (alpha_t / alpha_s0) * sample
                    - (sigma_t * (jnp.exp(h) - 1.0)) * D0
                    - (sigma_t * ((jnp.exp(h) - 1.0) / h - 1.0)) * D1
                )
        # 返回计算得到的x_t
        return x_t
    
    # 定义三阶更新的多步DPM求解器
    def multistep_dpm_solver_third_order_update(
        # 传入状态
        state: DPMSolverMultistepSchedulerState,
        # 传入模型输出列表
        model_output_list: jnp.ndarray,
        # 传入时间步列表
        timestep_list: List[int],
        # 传入前一个时间步
        prev_timestep: int,
        # 传入样本实例
        sample: jnp.ndarray,
    ) -> jnp.ndarray:  # 定义函数返回类型为 jnp.ndarray
        """  # 开始文档字符串
        One step for the third-order multistep DPM-Solver.  # 描述该函数为三阶多步 DPM 求解器的一步
        Args:  # 开始参数说明
            model_output_list (`List[jnp.ndarray]`):  # 定义模型输出列表参数
                direct outputs from learned diffusion model at current and latter timesteps.  # 描述该参数为当前及后续时间步的扩散模型直接输出
            timestep (`int`):  # 定义当前时间步参数
                current and latter discrete timestep in the diffusion chain.  # 描述该参数为扩散链中当前及后续离散时间步
            prev_timestep (`int`):  # 定义前一个时间步参数
                previous discrete timestep in the diffusion chain.  # 描述该参数为扩散链中前一个离散时间步
            sample (`jnp.ndarray`):  # 定义样本参数
                current instance of sample being created by diffusion process.  # 描述该参数为当前通过扩散过程创建的样本实例
        Returns:  # 开始返回值说明
            `jnp.ndarray`: the sample tensor at the previous timestep.  # 描述返回值为前一个时间步的样本张量
        """  # 结束文档字符串
        t, s0, s1, s2 = prev_timestep, timestep_list[-1], timestep_list[-2], timestep_list[-3]  # 获取当前和最近的四个时间步
        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]  # 获取最近三个模型输出
        lambda_t, lambda_s0, lambda_s1, lambda_s2 = (  # 从状态中提取对应时间步的 lambda 值
            state.lambda_t[t],  # 当前时间步的 lambda 值
            state.lambda_t[s0],  # 最近时间步的 lambda 值
            state.lambda_t[s1],  # 倒数第二个时间步的 lambda 值
            state.lambda_t[s2],  # 倒数第三个时间步的 lambda 值
        )  # 结束 lambda 值提取
        alpha_t, alpha_s0 = state.alpha_t[t], state.alpha_t[s0]  # 提取当前和最近时间步的 alpha 值
        sigma_t, sigma_s0 = state.sigma_t[t], state.sigma_t[s0]  # 提取当前和最近时间步的 sigma 值
        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2  # 计算 h 相关变量
        r0, r1 = h_0 / h, h_1 / h  # 计算 r0 和 r1
        D0 = m0  # 将最近的模型输出赋值给 D0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)  # 计算 D1_0 和 D1_1
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)  # 计算 D1
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)  # 计算 D2
        if self.config.algorithm_type == "dpmsolver++":  # 检查算法类型是否为 "dpmsolver++"
            # See https://arxiv.org/abs/2206.00927 for detailed derivations  # 引用文献以获取详细推导
            x_t = (  # 计算 x_t
                (sigma_t / sigma_s0) * sample  # 计算与 sigma_t 相关的项
                - (alpha_t * (jnp.exp(-h) - 1.0)) * D0  # 计算与 D0 相关的项
                + (alpha_t * ((jnp.exp(-h) - 1.0) / h + 1.0)) * D1  # 计算与 D1 相关的项
                - (alpha_t * ((jnp.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2  # 计算与 D2 相关的项
            )  # 结束 x_t 计算
        elif self.config.algorithm_type == "dpmsolver":  # 检查算法类型是否为 "dpmsolver"
            # See https://arxiv.org/abs/2206.00927 for detailed derivations  # 引用文献以获取详细推导
            x_t = (  # 计算 x_t
                (alpha_t / alpha_s0) * sample  # 计算与 alpha_t 相关的项
                - (sigma_t * (jnp.exp(h) - 1.0)) * D0  # 计算与 D0 相关的项
                - (sigma_t * ((jnp.exp(h) - 1.0) / h - 1.0)) * D1  # 计算与 D1 相关的项
                - (sigma_t * ((jnp.exp(h) - 1.0 - h) / h**2 - 0.5)) * D2  # 计算与 D2 相关的项
            )  # 结束 x_t 计算
        return x_t  # 返回计算出的 x_t
    def step(  # 定义 step 函数
        self,  # 传入自身引用
        state: DPMSolverMultistepSchedulerState,  # 定义状态参数
        model_output: jnp.ndarray,  # 定义模型输出参数
        timestep: int,  # 定义时间步参数
        sample: jnp.ndarray,  # 定义样本参数
        return_dict: bool = True,  # 定义是否返回字典的参数，默认为 True
    def scale_model_input(  # 定义 scale_model_input 函数
        self,  # 传入自身引用
        state: DPMSolverMultistepSchedulerState,  # 定义状态参数
        sample: jnp.ndarray,  # 定义样本参数
        timestep: Optional[int] = None  # 定义时间步参数，可选，默认为 None
    ) -> jnp.ndarray:  # 指定函数返回类型为 jnp.ndarray
        """  # 文档字符串，描述函数的作用及参数
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.  # 确保与需要根据当前时间步缩放去噪模型输入的调度器的可互换性

        Args:  # 参数说明部分
            state (`DPMSolverMultistepSchedulerState`):  # state 参数，类型为 DPMSolverMultistepSchedulerState
                the `FlaxDPMSolverMultistepScheduler` state data class instance.  # FlaxDPMSolverMultistepScheduler 的状态数据类实例
            sample (`jnp.ndarray`): input sample  # sample 参数，类型为 jnp.ndarray，表示输入样本
            timestep (`int`, optional): current timestep  # timestep 参数，类型为 int，可选，表示当前时间步

        Returns:  # 返回值说明部分
            `jnp.ndarray`: scaled input sample  # 返回一个 jnp.ndarray，表示缩放后的输入样本
        """
        return sample  # 返回输入样本，不做任何修改

    def add_noise(  # 定义 add_noise 函数
        self,  # 对象自身引用
        state: DPMSolverMultistepSchedulerState,  # state 参数，类型为 DPMSolverMultistepSchedulerState
        original_samples: jnp.ndarray,  # original_samples 参数，类型为 jnp.ndarray，表示原始样本
        noise: jnp.ndarray,  # noise 参数，类型为 jnp.ndarray，表示要添加的噪声
        timesteps: jnp.ndarray,  # timesteps 参数，类型为 jnp.ndarray，表示时间步
    ) -> jnp.ndarray:  # 指定函数返回类型为 jnp.ndarray
        return add_noise_common(state.common, original_samples, noise, timesteps)  # 调用 add_noise_common 函数，返回添加噪声后的样本

    def __len__(self):  # 定义获取对象长度的方法
        return self.config.num_train_timesteps  # 返回配置中定义的训练时间步数
```