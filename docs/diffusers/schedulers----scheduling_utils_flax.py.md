# `.\diffusers\schedulers\scheduling_utils_flax.py`

```py
# 版权声明，表明文件版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证第 2.0 版（"许可证"）进行许可；
# 除非遵循许可证，否则不得使用此文件。
# 可在以下网址获取许可证副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 以 "原样" 基础提供，不附带任何明示或暗示的担保或条件。
# 请参阅许可证以获取有关权限和
# 限制的具体语言。
import importlib  # 导入 importlib 模块，用于动态导入模块
import math  # 导入 math 模块，提供数学函数
import os  # 导入 os 模块，用于操作系统相关的功能
from dataclasses import dataclass  # 从 dataclasses 导入 dataclass，用于简化类定义
from enum import Enum  # 从 enum 导入 Enum，用于创建枚举类型
from typing import Optional, Tuple, Union  # 从 typing 导入类型提示工具

import flax  # 导入 flax 库，用于构建和训练神经网络
import jax.numpy as jnp  # 导入 jax.numpy 并重命名为 jnp，用于高性能数值计算
from huggingface_hub.utils import validate_hf_hub_args  # 从 huggingface_hub.utils 导入验证函数

from ..utils import BaseOutput, PushToHubMixin  # 从相对路径导入 BaseOutput 和 PushToHubMixin

SCHEDULER_CONFIG_NAME = "scheduler_config.json"  # 定义调度器配置文件的名称

# 注意：将此类型定义为枚举简化了文档中的使用，并防止
# 在调度器模块内使用时出现循环导入。
# 当作为管道中的类型使用时，它实际上是一个联合，因为实际的
# 调度器实例会传入。
class FlaxKarrasDiffusionSchedulers(Enum):
    FlaxDDIMScheduler = 1  # 定义 FlaxDDIM 调度器
    FlaxDDPMScheduler = 2  # 定义 FlaxDDPM 调度器
    FlaxPNDMScheduler = 3  # 定义 FlaxPNDM 调度器
    FlaxLMSDiscreteScheduler = 4  # 定义 FlaxLMS 离散调度器
    FlaxDPMSolverMultistepScheduler = 5  # 定义 FlaxDPM 多步求解器调度器
    FlaxEulerDiscreteScheduler = 6  # 定义 FlaxEuler 离散调度器

@dataclass
class FlaxSchedulerOutput(BaseOutput):
    """
    调度器步函数输出的基类。

    参数：
        prev_sample (`jnp.ndarray`，形状为 `(batch_size, num_channels, height, width)` 的图像)：
            计算的前一时间步样本 (x_{t-1})。`prev_sample` 应在
            去噪循环中用作下一个模型输入。
    """

    prev_sample: jnp.ndarray  # 定义前一时间步样本的属性

class FlaxSchedulerMixin(PushToHubMixin):
    """
    包含调度器通用功能的混入类。

    类属性：
        - **_compatibles** (`List[str]`) -- 兼容父类的类列表，以便
          可以从与保存配置不同的类使用 `from_config`（应由父类重写）。
    """

    config_name = SCHEDULER_CONFIG_NAME  # 设置配置名称属性
    ignore_for_config = ["dtype"]  # 定义在配置中忽略的属性
    _compatibles = []  # 初始化兼容类的空列表
    has_compatibles = True  # 设置兼容类存在的标志

    @classmethod
    @validate_hf_hub_args  # 应用装饰器以验证 Hugging Face Hub 参数
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,  # 预训练模型名称或路径
        subfolder: Optional[str] = None,  # 子文件夹参数
        return_unused_kwargs=False,  # 是否返回未使用的关键字参数
        **kwargs,  # 其他可选关键字参数
    # 定义一个保存预训练模型配置的方法，接受保存目录和其他参数
    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        # 保存调度器配置对象到指定目录，以便后续可以使用从预训练加载方法重新加载
        """
        Save a scheduler configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~FlaxSchedulerMixin.from_pretrained`] class method.
    
        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        # 调用保存配置的方法，传入保存目录、是否推送到 Hub 及其他参数
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
    
    # 定义一个属性，返回与当前调度器兼容的所有调度器
    @property
    def compatibles(self):
        # 返回与该调度器兼容的调度器列表
        """
        Returns all schedulers that are compatible with this scheduler
    
        Returns:
            `List[SchedulerMixin]`: List of compatible schedulers
        """
        # 调用获取兼容调度器的方法并返回结果
        return self._get_compatibles()
    
    # 定义一个类方法，用于获取兼容的调度器类
    @classmethod
    def _get_compatibles(cls):
        # 获取当前类名及其兼容类名的集合
        compatible_classes_str = list(set([cls.__name__] + cls._compatibles))
        # 导入当前模块的主库
        diffusers_library = importlib.import_module(__name__.split(".")[0])
        # 遍历兼容类名列表，获取存在于库中的类
        compatible_classes = [
            getattr(diffusers_library, c) for c in compatible_classes_str if hasattr(diffusers_library, c)
        ]
        # 返回兼容类列表
        return compatible_classes
# 定义一个函数，用于从左侧广播数组到指定形状
def broadcast_to_shape_from_left(x: jnp.ndarray, shape: Tuple[int]) -> jnp.ndarray:
    # 断言目标形状的维度不小于输入数组的维度
    assert len(shape) >= x.ndim
    # 将输入数组重塑为目标形状，并进行广播
    return jnp.broadcast_to(x.reshape(x.shape + (1,) * (len(shape) - x.ndim)), shape)

# 定义一个函数，生成beta调度以离散化给定的alpha_t_bar函数
def betas_for_alpha_bar(num_diffusion_timesteps: int, max_beta=0.999, dtype=jnp.float32) -> jnp.ndarray:
    """
    创建一个beta调度，离散化给定的alpha_t_bar函数，定义(1-beta)的累积乘积。

    包含一个alpha_bar函数，该函数将参数t转化为(1-beta)的累积乘积。

    Args:
        num_diffusion_timesteps (`int`): 生成的beta数量。
        max_beta (`float`): 最大beta值；使用小于1的值以避免奇异性。

    Returns:
        betas (`jnp.ndarray`): 调度器用来更新模型输出的betas
    """

    # 定义一个内部函数，计算给定时间步的alpha_bar值
    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    # 初始化betas列表
    betas = []
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前和下一个时间步
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        # 将计算出的beta添加到列表中，限制其最大值
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    # 返回beta数组
    return jnp.array(betas, dtype=dtype)

# 定义一个数据类，用于表示通用调度器的状态
@flax.struct.dataclass
class CommonSchedulerState:
    alphas: jnp.ndarray
    betas: jnp.ndarray
    alphas_cumprod: jnp.ndarray

    # 定义一个类方法，用于创建调度器状态
    @classmethod
    def create(cls, scheduler):
        config = scheduler.config

        # 检查配置中是否有训练好的betas
        if config.trained_betas is not None:
            # 将训练好的betas转换为数组
            betas = jnp.asarray(config.trained_betas, dtype=scheduler.dtype)
        # 根据配置选择不同的beta调度
        elif config.beta_schedule == "linear":
            # 创建线性beta调度
            betas = jnp.linspace(config.beta_start, config.beta_end, config.num_train_timesteps, dtype=scheduler.dtype)
        elif config.beta_schedule == "scaled_linear":
            # 创建特定于潜在扩散模型的调度
            betas = (
                jnp.linspace(
                    config.beta_start**0.5, config.beta_end**0.5, config.num_train_timesteps, dtype=scheduler.dtype
                )
                ** 2
            )
        elif config.beta_schedule == "squaredcos_cap_v2":
            # 使用Glide余弦调度
            betas = betas_for_alpha_bar(config.num_train_timesteps, dtype=scheduler.dtype)
        else:
            # 如果beta调度未实现，则抛出错误
            raise NotImplementedError(
                f"beta_schedule {config.beta_schedule} is not implemented for scheduler {scheduler.__class__.__name__}"
            )

        # 计算alphas为1减去betas
        alphas = 1.0 - betas
        # 计算累积的alphas乘积
        alphas_cumprod = jnp.cumprod(alphas, axis=0)

        # 返回包含alphas、betas和alphas_cumprod的调度器状态
        return cls(
            alphas=alphas,
            betas=betas,
            alphas_cumprod=alphas_cumprod,
        )

# 定义一个函数，获取平方根的alphas乘积
def get_sqrt_alpha_prod(
    state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray
):
    # 获取当前状态的alphas累积乘积
    alphas_cumprod = state.alphas_cumprod
    # 计算对应时间步的平方根alphas乘积
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    # 将 sqrt_alpha_prod 转换为一维数组
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 将一维数组根据原始样本形状从左侧广播扩展
        sqrt_alpha_prod = broadcast_to_shape_from_left(sqrt_alpha_prod, original_samples.shape)
    
        # 计算 1 减去 alpha 累积乘积的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 转换为一维数组
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 将一维数组根据原始样本形状从左侧广播扩展
        sqrt_one_minus_alpha_prod = broadcast_to_shape_from_left(sqrt_one_minus_alpha_prod, original_samples.shape)
    
        # 返回 sqrt_alpha_prod 和 sqrt_one_minus_alpha_prod 两个结果
        return sqrt_alpha_prod, sqrt_one_minus_alpha_prod
# 定义一个函数，添加噪声到原始样本
def add_noise_common(
    # 接受调度状态、原始样本、噪声和时间步长作为参数
    state: CommonSchedulerState, original_samples: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray
):
    # 计算平方根的 alpha 和 (1-alpha) 的乘积
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state, original_samples, noise, timesteps)
    # 生成带噪声的样本，结合原始样本和噪声
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    # 返回带噪声的样本
    return noisy_samples


# 定义一个函数，计算样本的速度
def get_velocity_common(state: CommonSchedulerState, sample: jnp.ndarray, noise: jnp.ndarray, timesteps: jnp.ndarray):
    # 计算平方根的 alpha 和 (1-alpha) 的乘积
    sqrt_alpha_prod, sqrt_one_minus_alpha_prod = get_sqrt_alpha_prod(state, sample, noise, timesteps)
    # 根据噪声和样本计算速度
    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    # 返回计算得到的速度
    return velocity
```