# `.\diffusers\schedulers\scheduling_karras_ve_flax.py`

```py
# 版权所有 2024 NVIDIA 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵循该许可证，否则不得使用此文件。
# 你可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件在“按原样”基础上分发，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证下特定权限和限制，请参见许可证。

# 从 dataclasses 模块导入 dataclass 装饰器，用于创建数据类
from dataclasses import dataclass
# 从 typing 模块导入 Optional、Tuple 和 Union 类型提示
from typing import Optional, Tuple, Union

# 导入 flax 库
import flax
# 导入 jax 库
import jax
# 导入 jax.numpy 模块，并命名为 jnp
import jax.numpy as jnp
# 从 jax 库导入 random 模块
from jax import random

# 从配置工具模块导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入 BaseOutput 类
from ..utils import BaseOutput
# 从调度工具模块导入 FlaxSchedulerMixin 类
from .scheduling_utils_flax import FlaxSchedulerMixin

# 使用 flax 库的 struct.dataclass 装饰器定义 KarrasVeSchedulerState 数据类
@flax.struct.dataclass
class KarrasVeSchedulerState:
    # 可设置的值
    num_inference_steps: Optional[int] = None  # 推理步骤数量，默认为 None
    timesteps: Optional[jnp.ndarray] = None  # 时间步数组，默认为 None
    schedule: Optional[jnp.ndarray] = None  # 调度数组，表示 sigma(t_i)，默认为 None

    # 类方法，用于创建 KarrasVeSchedulerState 的实例
    @classmethod
    def create(cls):
        return cls()  # 返回类的实例

# 使用 dataclass 装饰器定义 FlaxKarrasVeOutput 数据类
@dataclass
class FlaxKarrasVeOutput(BaseOutput):
    """
    调度器的步骤函数输出的输出类。

    参数:
        prev_sample (`jnp.ndarray`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            计算的样本 (x_{t-1})，表示前一个时间步的样本。`prev_sample` 应作为下一个模型输入
            在去噪循环中使用。
        derivative (`jnp.ndarray`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            预测原始图像样本 (x_0) 的导数。
        state (`KarrasVeSchedulerState`): `FlaxKarrasVeScheduler` 的状态数据类。
    """

    prev_sample: jnp.ndarray  # 前一个时间步的样本
    derivative: jnp.ndarray  # 预测原始样本的导数
    state: KarrasVeSchedulerState  # 调度器状态

# 定义 FlaxKarrasVeScheduler 类，继承自 FlaxSchedulerMixin 和 ConfigMixin
class FlaxKarrasVeScheduler(FlaxSchedulerMixin, ConfigMixin):
    """
    从 Karras 等人 [1] 定制的变异扩展 (VE) 模型进行随机采样。请参考 [1] 的算法 2 和表 1 中的 VE 列。

    [1] Karras, Tero 等人。 "阐明基于扩散的生成模型的设计空间。"
    https://arxiv.org/abs/2206.00364 [2] Song, Yang 等人。 "通过随机微分方程进行基于得分的生成建模。" 
    https://arxiv.org/abs/2011.13456

    [`~ConfigMixin`] 处理存储在调度器 `__init__` 函数中传入的所有配置属性，
    如 `num_train_timesteps`。它们可以通过 `scheduler.config.num_train_timesteps` 访问。
    [`SchedulerMixin`] 通过 [`SchedulerMixin.save_pretrained`] 和
    [`~SchedulerMixin.from_pretrained`] 函数提供一般的加载和保存功能。

    有关参数的更多详细信息，请参见原始论文的附录 E：“阐明设计空间的扩散模型
    # 引用论文的标题和链接，说明相关模型的研究
    Diffusion-Based Generative Models." https://arxiv.org/abs/2206.00364. # 说明用于查找特定模型的最佳参数的表格位置
    The grid search values used to find the
    # 定义参数说明
    optimal {s_noise, s_churn, s_min, s_max} for a specific model are described in Table 5 of the paper.

    # 参数 sigma_min：最小噪声强度
    Args:
        sigma_min (`float`): minimum noise magnitude
        # 参数 sigma_max：最大噪声强度
        sigma_max (`float`): maximum noise magnitude
        # 参数 s_noise：在采样过程中抵消细节丢失的额外噪声量，合理范围为 [1.000, 1.011]
        s_noise (`float`): the amount of additional noise to counteract loss of detail during sampling.
            A reasonable range is [1.000, 1.011].
        # 参数 s_churn：控制整体随机性的参数，合理范围为 [0, 100]
        s_churn (`float`): the parameter controlling the overall amount of stochasticity.
            A reasonable range is [0, 100].
        # 参数 s_min：开始添加噪声的 sigma 范围起始值，合理范围为 [0, 10]
        s_min (`float`): the start value of the sigma range where we add noise (enable stochasticity).
            A reasonable range is [0, 10].
        # 参数 s_max：添加噪声的 sigma 范围结束值，合理范围为 [0.2, 80]
        s_max (`float`): the end value of the sigma range where we add noise.
            A reasonable range is [0.2, 80].
    # 文档字符串结束
    """

    # 属性 has_state：返回布尔值 True，表示存在状态
    @property
    def has_state(self):
        return True

    # 注册初始化方法，接收多个噪声参数，并设置默认值
    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.02, # 初始化时设置最小噪声强度的默认值
        sigma_max: float = 100,   # 初始化时设置最大噪声强度的默认值
        s_noise: float = 1.007,   # 初始化时设置额外噪声的默认值
        s_churn: float = 80,      # 初始化时设置整体随机性的默认值
        s_min: float = 0.05,      # 初始化时设置 sigma 范围起始值的默认值
        s_max: float = 50,        # 初始化时设置 sigma 范围结束值的默认值
    ):
        # 方法体为空，表示初始化时无额外操作
        pass

    # 创建并返回 KarrasVeSchedulerState 状态实例
    def create_state(self):
        return KarrasVeSchedulerState.create()

    # 设置扩散链使用的连续时间步，支持推理前运行
    def set_timesteps(
        self, state: KarrasVeSchedulerState, num_inference_steps: int, shape: Tuple = ()
    ) -> KarrasVeSchedulerState:
        """
        # 定义时间步设置的文档字符串，包含参数说明
        Sets the continuous timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            state (`KarrasVeSchedulerState`):
                the `FlaxKarrasVeScheduler` state data class.
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.

        """
        # 生成从 num_inference_steps 到 0 的时间步数组
        timesteps = jnp.arange(0, num_inference_steps)[::-1].copy()
        # 根据公式计算时间步的调度
        schedule = [
            (
                self.config.sigma_max**2
                * (self.config.sigma_min**2 / self.config.sigma_max**2) ** (i / (num_inference_steps - 1))
            )
            for i in timesteps
        ]

        # 替换状态，更新时间步数、调度和时间步数组
        return state.replace(
            num_inference_steps=num_inference_steps,
            schedule=jnp.array(schedule, dtype=jnp.float32),
            timesteps=timesteps,
        )

    # 添加噪声到输入样本的方法
    def add_noise_to_input(
        self,
        state: KarrasVeSchedulerState,  # 当前状态
        sample: jnp.ndarray,            # 输入样本数据
        sigma: float,                   # 当前噪声强度
        key: jax.Array,                 # 随机数生成的键值
    ) -> Tuple[jnp.ndarray, float]:  # 定义函数返回一个元组，其中第一个元素是 jnp.ndarray 类型，第二个元素是 float 类型
        """
        Explicit Langevin-like "churn" step of adding noise to the sample according to a factor gamma_i ≥ 0 to reach a
        higher noise level sigma_hat = sigma_i + gamma_i*sigma_i.  # 文档字符串，描述了函数的作用和参数

        TODO Args:  # TODO: 这里是待完成的参数说明部分
        """
        if self.config.s_min <= sigma <= self.config.s_max:  # 检查 sigma 是否在配置的最小和最大值之间
            gamma = min(self.config.s_churn / state.num_inference_steps, 2**0.5 - 1)  # 计算 gamma 值，限制其最大值
        else:  # 如果 sigma 不在范围内
            gamma = 0  # 将 gamma 设置为 0

        # sample eps ~ N(0, S_noise^2 * I)  # 从正态分布中采样噪声 eps
        key = random.split(key, num=1)  # 将随机数生成器的 key 分裂，得到新的 key
        eps = self.config.s_noise * random.normal(key=key, shape=sample.shape)  # 根据配置的噪声标准差生成噪声
        sigma_hat = sigma + gamma * sigma  # 计算新的噪声水平 sigma_hat
        sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)  # 更新样本，加上噪声

        return sample_hat, sigma_hat  # 返回更新后的样本和新的噪声水平

    def step(  # 定义 step 方法
        self,
        state: KarrasVeSchedulerState,  # 当前状态对象，包含调度器的状态
        model_output: jnp.ndarray,  # 从学习到的扩散模型输出的张量
        sigma_hat: float,  # 当前的噪声水平
        sigma_prev: float,  # 上一个噪声水平
        sample_hat: jnp.ndarray,  # 更新后的样本
        return_dict: bool = True,  # 返回结果的类型，默认为 True
    ) -> Union[FlaxKarrasVeOutput, Tuple]:  # 返回类型可以是 FlaxKarrasVeOutput 或元组
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).  # 文档字符串，描述了函数的功能

        Args:  # 参数说明部分
            state (`KarrasVeSchedulerState`): the `FlaxKarrasVeScheduler` state data class.  # 状态对象
            model_output (`torch.Tensor` or `np.ndarray`): direct output from learned diffusion model.  # 模型输出
            sigma_hat (`float`): TODO  # 待完成的参数说明
            sigma_prev (`float`): TODO  # 待完成的参数说明
            sample_hat (`torch.Tensor` or `np.ndarray`): TODO  # 待完成的参数说明
            return_dict (`bool`): option for returning tuple rather than FlaxKarrasVeOutput class  # 返回类型选项

        Returns:  # 返回值说明
            [`~schedulers.scheduling_karras_ve_flax.FlaxKarrasVeOutput`] or `tuple`: Updated sample in the diffusion
            chain and derivative. [`~schedulers.scheduling_karras_ve_flax.FlaxKarrasVeOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is the sample tensor.  # 返回值的描述
        """

        pred_original_sample = sample_hat + sigma_hat * model_output  # 预测原始样本，结合当前样本和模型输出
        derivative = (sample_hat - pred_original_sample) / sigma_hat  # 计算样本与预测样本之间的导数
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative  # 计算上一个样本

        if not return_dict:  # 如果不返回字典
            return (sample_prev, derivative, state)  # 返回样本、导数和状态

        return FlaxKarrasVeOutput(prev_sample=sample_prev, derivative=derivative, state=state)  # 返回结果对象

    def step_correct(  # 定义 step_correct 方法
        self,
        state: KarrasVeSchedulerState,  # 当前状态对象
        model_output: jnp.ndarray,  # 模型输出的张量
        sigma_hat: float,  # 当前噪声水平
        sigma_prev: float,  # 上一个噪声水平
        sample_hat: jnp.ndarray,  # 更新后的样本
        sample_prev: jnp.ndarray,  # 上一个样本
        derivative: jnp.ndarray,  # 样本的导数
        return_dict: bool = True,  # 返回类型选项
    ) -> Union[FlaxKarrasVeOutput, Tuple]:
        """
        修正预测的样本，基于网络的输出 model_output。TODO 完成描述

        参数:
            state (`KarrasVeSchedulerState`): `FlaxKarrasVeScheduler` 的状态数据类。
            model_output (`torch.Tensor` 或 `np.ndarray`): 从学习的扩散模型直接输出。
            sigma_hat (`float`): TODO
            sigma_prev (`float`): TODO
            sample_hat (`torch.Tensor` 或 `np.ndarray`): TODO
            sample_prev (`torch.Tensor` 或 `np.ndarray`): TODO
            derivative (`torch.Tensor` 或 `np.ndarray`): TODO
            return_dict (`bool`): 返回元组而不是 FlaxKarrasVeOutput 类的选项

        返回:
            prev_sample (TODO): 扩散链中更新的样本。derivative (TODO): TODO

        """
        # 计算原始预测样本，基于前一个样本和模型输出的缩放
        pred_original_sample = sample_prev + sigma_prev * model_output
        # 计算修正后的导数，使用当前样本与预测样本的差值
        derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
        # 更新前一个样本，考虑新的样本和导数的影响
        sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)

        # 检查是否返回字典形式的结果
        if not return_dict:
            # 返回样本、导数和状态的元组
            return (sample_prev, derivative, state)

        # 返回 FlaxKarrasVeOutput 对象，包含更新的样本、导数和状态
        return FlaxKarrasVeOutput(prev_sample=sample_prev, derivative=derivative, state=state)

    # 定义一个添加噪声的方法，尚未实现
    def add_noise(self, state: KarrasVeSchedulerState, original_samples, noise, timesteps):
        # 抛出未实现错误
        raise NotImplementedError()
```