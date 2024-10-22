# `.\diffusers\schedulers\deprecated\scheduling_karras_ve.py`

```py
# 版权所有 2024 NVIDIA 和 The HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（"许可证"）许可；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，软件在许可证下分发是以“按原样”基础进行的，
# 不提供任何形式的明示或暗示的担保或条件。
# 有关许可证下权限和限制的具体语言，请参见许可证。


# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入可选类型、元组和联合类型
from typing import Optional, Tuple, Union

# 导入 NumPy 库，通常用于数值计算
import numpy as np
# 导入 PyTorch 库，通常用于深度学习
import torch

# 从配置工具中导入 ConfigMixin 和 register_to_config
from ...configuration_utils import ConfigMixin, register_to_config
# 从 utils 模块导入 BaseOutput 基类
from ...utils import BaseOutput
# 从 utils.torch_utils 导入生成随机张量的函数
from ...utils.torch_utils import randn_tensor
# 从调度工具中导入 SchedulerMixin
from ..scheduling_utils import SchedulerMixin


# 定义 KarrasVeOutput 类，继承自 BaseOutput
@dataclass
class KarrasVeOutput(BaseOutput):
    """
    调度器步骤函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像)：
            先前时间步的计算样本 (x_{t-1})。`prev_sample` 应作为下一个模型输入使用
            在去噪循环中。
        derivative (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像)：
            预测的原始图像样本的导数 (x_0)。
        pred_original_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像)：
            基于当前时间步模型输出的预测去噪样本 (x_{0})。
            `pred_original_sample` 可用于预览进度或进行引导。
    """

    # 先前样本，类型为 torch.Tensor
    prev_sample: torch.Tensor
    # 导数，类型为 torch.Tensor
    derivative: torch.Tensor
    # 可选的预测原始样本，类型为 torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


# 定义 KarrasVeScheduler 类，继承自 SchedulerMixin 和 ConfigMixin
class KarrasVeScheduler(SchedulerMixin, ConfigMixin):
    """
    针对方差扩展模型的随机调度器。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关库为所有调度器实现的通用
    方法的详细信息，请查看超类文档，例如加载和保存。

    <Tip>

    有关参数的更多详细信息，请参见 [附录 E](https://arxiv.org/abs/2206.00364)。用于查找特定模型的
    最优 `{s_noise, s_churn, s_min, s_max}` 的网格搜索值在论文的表 5 中进行了描述。

    </Tip>
    # 参数说明部分，描述每个参数的含义和默认值
        Args:
            sigma_min (`float`, defaults to 0.02):
                # 最小噪声幅度
                The minimum noise magnitude.
            sigma_max (`float`, defaults to 100):
                # 最大噪声幅度
                The maximum noise magnitude.
            s_noise (`float`, defaults to 1.007):
                # 额外噪声量，抵消采样时的细节损失，合理范围为 [1.000, 1.011]
                The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
                1.011].
            s_churn (`float`, defaults to 80):
                # 控制整体随机性程度的参数，合理范围为 [0, 100]
                The parameter controlling the overall amount of stochasticity. A reasonable range is [0, 100].
            s_min (`float`, defaults to 0.05):
                # 添加噪声的起始 sigma 范围值，合理范围为 [0, 10]
                The start value of the sigma range to add noise (enable stochasticity). A reasonable range is [0, 10].
            s_max (`float`, defaults to 50):
                # 添加噪声的结束 sigma 范围值，合理范围为 [0.2, 80]
                The end value of the sigma range to add noise. A reasonable range is [0.2, 80].
        """
    
        # 定义阶数为 2
        order = 2
    
        # 初始化方法，注册到配置
        @register_to_config
        def __init__(
            self,
            # 最小噪声幅度，默认值为 0.02
            sigma_min: float = 0.02,
            # 最大噪声幅度，默认值为 100
            sigma_max: float = 100,
            # 额外噪声量，默认值为 1.007
            s_noise: float = 1.007,
            # 随机性控制参数，默认值为 80
            s_churn: float = 80,
            # sigma 范围起始值，默认值为 0.05
            s_min: float = 0.05,
            # sigma 范围结束值，默认值为 50
            s_max: float = 50,
        ):
            # 设置初始噪声分布的标准差
            self.init_noise_sigma = sigma_max
    
            # 可设置值
            # 推理步骤的数量，初始为 None
            self.num_inference_steps: int = None
            # 时间步的张量，初始为 None
            self.timesteps: np.IntTensor = None
            # sigma(t_i) 的张量，初始为 None
            self.schedule: torch.Tensor = None  # sigma(t_i)
    
        # 处理模型输入以确保与调度器的互换性
        def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
            """
            确保与需要根据当前时间步缩放去噪模型输入的调度器的互换性。
    
            Args:
                sample (`torch.Tensor`):
                    # 输入样本
                    The input sample.
                timestep (`int`, *optional*):
                    # 当前扩散链中的时间步
                    The current timestep in the diffusion chain.
    
            Returns:
                `torch.Tensor`:
                    # 返回缩放后的输入样本
                    A scaled input sample.
            """
            # 返回未改变的样本
            return sample
    
        # 设置扩散链使用的离散时间步
        def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
            """
            设置用于扩散链的离散时间步（在推理之前运行）。
    
            Args:
                num_inference_steps (`int`):
                    # 生成样本时使用的扩散步骤数量
                    The number of diffusion steps used when generating samples with a pre-trained model.
                device (`str` or `torch.device`, *optional*):
                    # 将时间步移动到的设备，如果为 None，则不移动
                    The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            """
            # 设置推理步骤数量
            self.num_inference_steps = num_inference_steps
            # 创建时间步数组并反转
            timesteps = np.arange(0, self.num_inference_steps)[::-1].copy()
            # 将时间步转换为张量并移动到指定设备
            self.timesteps = torch.from_numpy(timesteps).to(device)
            # 计算调度的 sigma 值
            schedule = [
                (
                    self.config.sigma_max**2
                    * (self.config.sigma_min**2 / self.config.sigma_max**2) ** (i / (num_inference_steps - 1))
                )
                for i in self.timesteps
            ]
            # 将调度值转换为张量
            self.schedule = torch.tensor(schedule, dtype=torch.float32, device=device)
    # 定义添加噪声到输入样本的函数
    def add_noise_to_input(
            self, sample: torch.Tensor, sigma: float, generator: Optional[torch.Generator] = None
        ) -> Tuple[torch.Tensor, float]:
            """
            显式的 Langevin 类似的“搅动”步骤，根据 `gamma_i ≥ 0` 添加噪声，以达到更高的噪声水平 `sigma_hat = sigma_i + gamma_i*sigma_i`。
    
            参数:
                sample (`torch.Tensor`):
                    输入样本。
                sigma (`float`):
                generator (`torch.Generator`, *可选*):
                    随机数生成器。
            """
            # 检查 sigma 是否在配置的最小值和最大值之间
            if self.config.s_min <= sigma <= self.config.s_max:
                # 计算 gamma，确保不会超过最大值
                gamma = min(self.config.s_churn / self.num_inference_steps, 2**0.5 - 1)
            else:
                # 如果不在范围内，gamma 为 0
                gamma = 0
    
            # 从标准正态分布中采样噪声 eps
            eps = self.config.s_noise * randn_tensor(sample.shape, generator=generator).to(sample.device)
            # 计算新的噪声水平
            sigma_hat = sigma + gamma * sigma
            # 更新样本，添加噪声
            sample_hat = sample + ((sigma_hat**2 - sigma**2) ** 0.5 * eps)
    
            # 返回更新后的样本和新的噪声水平
            return sample_hat, sigma_hat
    
    # 定义从上一个时间步预测样本的步骤函数
    def step(
            self,
            model_output: torch.Tensor,
            sigma_hat: float,
            sigma_prev: float,
            sample_hat: torch.Tensor,
            return_dict: bool = True,
        ) -> Union[KarrasVeOutput, Tuple]:
            """
            通过反转 SDE 从学习的模型输出中传播扩散过程（通常是预测的噪声）。
    
            参数:
                model_output (`torch.Tensor`):
                    学习扩散模型的直接输出。
                sigma_hat (`float`):
                sigma_prev (`float`):
                sample_hat (`torch.Tensor`):
                return_dict (`bool`, *可选*, 默认为 `True`):
                    是否返回一个 [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`] 或 `tuple`。
    
            返回:
                [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`] 或 `tuple`:
                    如果 return_dict 为 `True`，返回 [`~schedulers.scheduling_karras_ve.KarrasVESchedulerOutput`]，
                    否则返回一个元组，第一个元素是样本张量。
            """
    
            # 根据模型输出和 sigma_hat 计算预测的原始样本
            pred_original_sample = sample_hat + sigma_hat * model_output
            # 计算样本的导数
            derivative = (sample_hat - pred_original_sample) / sigma_hat
            # 计算上一个时间步的样本
            sample_prev = sample_hat + (sigma_prev - sigma_hat) * derivative
    
            # 如果不返回字典，返回样本和导数
            if not return_dict:
                return (sample_prev, derivative)
    
            # 返回包含样本、导数和预测原始样本的 KarrasVeOutput 对象
            return KarrasVeOutput(
                prev_sample=sample_prev, derivative=derivative, pred_original_sample=pred_original_sample
            )
    
    # 定义带有修正步骤的函数
    def step_correct(
            self,
            model_output: torch.Tensor,
            sigma_hat: float,
            sigma_prev: float,
            sample_hat: torch.Tensor,
            sample_prev: torch.Tensor,
            derivative: torch.Tensor,
            return_dict: bool = True,
    # 处理网络的模型输出，纠正预测样本
    ) -> Union[KarrasVeOutput, Tuple]:
            """
            # 根据网络的模型输出修正预测样本
    
            Args:
                model_output (`torch.Tensor`):
                    # 从学习的扩散模型直接输出的张量
                sigma_hat (`float`): TODO
                sigma_prev (`float`): TODO
                sample_hat (`torch.Tensor`): TODO
                sample_prev (`torch.Tensor`): TODO
                derivative (`torch.Tensor`): TODO
                return_dict (`bool`, *optional*, defaults to `True`):
                    # 是否返回 [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] 或 `tuple`
    
            Returns:
                prev_sample (TODO): # 在扩散链中的更新样本。 derivative (TODO): TODO
    
            """
            # 通过前一个样本和模型输出计算预测的原始样本
            pred_original_sample = sample_prev + sigma_prev * model_output
            # 计算修正后的导数
            derivative_corr = (sample_prev - pred_original_sample) / sigma_prev
            # 更新前一个样本，根据当前和预测的导数
            sample_prev = sample_hat + (sigma_prev - sigma_hat) * (0.5 * derivative + 0.5 * derivative_corr)
    
            # 如果不返回字典，则返回更新的样本和导数
            if not return_dict:
                return (sample_prev, derivative)
    
            # 返回 KarrasVeOutput 对象，包含更新的样本和导数
            return KarrasVeOutput(
                prev_sample=sample_prev, derivative=derivative, pred_original_sample=pred_original_sample
            )
    
        # 声明未实现的方法，用于添加噪声
        def add_noise(self, original_samples, noise, timesteps):
            # 引发未实现错误
            raise NotImplementedError()
```