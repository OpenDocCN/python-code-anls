# `.\diffusers\schedulers\scheduling_sde_ve.py`

```py
# 版权所有 2024 Google Brain 和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有规定，否则根据许可证分发的软件按“现状”提供，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 有关许可证具体条款的权限和限制，请参阅许可证。

# 免责声明：此文件受到 https://github.com/yang-song/score_sde_pytorch 的强烈影响

# 导入数学库，用于数学运算
import math
# 从 dataclasses 模块导入 dataclass，用于创建数据类
from dataclasses import dataclass
# 从 typing 模块导入 Optional、Tuple 和 Union，用于类型注释
from typing import Optional, Tuple, Union

# 导入 PyTorch 库，用于深度学习模型的构建和训练
import torch

# 导入配置工具类和注册配置装饰器
from ..configuration_utils import ConfigMixin, register_to_config
# 导入基本输出类
from ..utils import BaseOutput
# 导入生成随机张量的工具函数
from ..utils.torch_utils import randn_tensor
# 导入调度相关的工具类
from .scheduling_utils import SchedulerMixin, SchedulerOutput

# 定义一个数据类，表示调度器的输出
@dataclass
class SdeVeOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            上一时间步的计算样本 `(x_{t-1})`。`prev_sample` 应作为去噪循环中的下一个模型输入。
        prev_sample_mean (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            在前一时间步平均的 `prev_sample`。
    """

    # 定义上一时间步的样本张量
    prev_sample: torch.Tensor
    # 定义上一时间步样本的平均值张量
    prev_sample_mean: torch.Tensor


# 定义一个类，表示方差爆炸的随机微分方程调度器
class ScoreSdeVeScheduler(SchedulerMixin, ConfigMixin):
    """
    `ScoreSdeVeScheduler` 是一个方差爆炸的随机微分方程 (SDE) 调度器。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查看超类文档以获取库为所有调度器实现的通用方法，例如加载和保存。

    参数：
        num_train_timesteps (`int`，默认值为 1000):
            用于训练模型的扩散步骤数。
        snr (`float`，默认值为 0.15):
            权衡来自 `model_output` 样本（来自网络）到随机噪声的步骤系数。
        sigma_min (`float`，默认值为 0.01):
            采样过程中 sigma 序列的初始噪声尺度。最小 sigma 应该反映数据的分布。
        sigma_max (`float`，默认值为 1348.0):
            用于传入模型的连续时间步范围的最大值。
        sampling_eps (`float`，默认值为 1e-5):
            采样结束值，时间步逐渐从 1 减少到 epsilon。
        correct_steps (`int`，默认值为 1):
            对生成样本执行的校正步骤数量。
    """

    # 定义调度器的顺序
    order = 1

    # 将配置注册到调度器
    @register_to_config
    # 初始化类的构造函数
    def __init__(
            # 训练时间步的数量，默认为2000
            num_train_timesteps: int = 2000,
            # 信噪比，默认为0.15
            snr: float = 0.15,
            # 最小噪声标准差，默认为0.01
            sigma_min: float = 0.01,
            # 最大噪声标准差，默认为1348.0
            sigma_max: float = 1348.0,
            # 采样的最小值，默认为1e-5
            sampling_eps: float = 1e-5,
            # 校正步数，默认为1
            correct_steps: int = 1,
        ):
            # 设置初始噪声分布的标准差为最大值
            self.init_noise_sigma = sigma_max
    
            # 可设置的时间步值，初始为None
            self.timesteps = None
    
            # 设置噪声标准差
            self.set_sigmas(num_train_timesteps, sigma_min, sigma_max, sampling_eps)
    
    # 输入样本和时间步的缩放函数
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
            """
            确保可以与调度器兼容，根据当前时间步缩放去噪模型的输入。
    
            参数：
                sample (`torch.Tensor`):
                    输入样本。
                timestep (`int`, *可选*):
                    扩散链中的当前时间步。
    
            返回：
                `torch.Tensor`:
                    一个缩放后的输入样本。
            """
            # 返回未缩放的样本
            return sample
    
    # 设置扩散链的连续时间步
    def set_timesteps(
            # 推断步骤的数量
            self, num_inference_steps: int, sampling_eps: float = None, device: Union[str, torch.device] = None
        ):
            """
            在推断前设置用于扩散链的连续时间步。
    
            参数：
                num_inference_steps (`int`):
                    生成样本时使用的扩散步骤数量。
                sampling_eps (`float`, *可选*):
                    最终时间步值（覆盖调度器实例化时提供的值）。
                device (`str` 或 `torch.device`, *可选*):
                    时间步要移动到的设备。如果为`None`，则不移动。
    
            """
            # 设置采样最小值，如果未提供则使用配置中的值
            sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
    
            # 创建等间隔的时间步，并将其放在指定设备上
            self.timesteps = torch.linspace(1, sampling_eps, num_inference_steps, device=device)
    
    # 设置噪声标准差的函数
    def set_sigmas(
            # 推断步骤的数量
            self, num_inference_steps: int, sigma_min: float = None, sigma_max: float = None, sampling_eps: float = None
    ):
        """
        设置扩散链中使用的噪声尺度（在推理之前运行）。sigmas 控制样本更新中 `drift` 和 `diffusion` 组件的权重。

        参数：
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数量，使用预训练模型。
            sigma_min (`float`, optional):
                初始噪声尺度值（覆盖调度器实例化时给定的值）。
            sigma_max (`float`, optional):
                最终噪声尺度值（覆盖调度器实例化时给定的值）。
            sampling_eps (`float`, optional):
                最终时间步值（覆盖调度器实例化时给定的值）。
        """
        # 如果 sigma_min 为 None，则使用配置中的 sigma_min
        sigma_min = sigma_min if sigma_min is not None else self.config.sigma_min
        # 如果 sigma_max 为 None，则使用配置中的 sigma_max
        sigma_max = sigma_max if sigma_max is not None else self.config.sigma_max
        # 如果 sampling_eps 为 None，则使用配置中的 sampling_eps
        sampling_eps = sampling_eps if sampling_eps is not None else self.config.sampling_eps
        # 如果 timesteps 为 None，则根据推理步骤和采样 epsilon 设置时间步
        if self.timesteps is None:
            self.set_timesteps(num_inference_steps, sampling_eps)

        # 计算 sigmas，依据 sigma_min 和 sigma_max 以及时间步
        self.sigmas = sigma_min * (sigma_max / sigma_min) ** (self.timesteps / sampling_eps)
        # 生成离散的 sigmas，通过指数函数计算从 sigma_min 到 sigma_max 的数值
        self.discrete_sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_max), num_inference_steps))
        # 将 sigmas 转换为张量形式，依据时间步进行计算
        self.sigmas = torch.tensor([sigma_min * (sigma_max / sigma_min) ** t for t in self.timesteps])

    # 获取与当前时间步相邻的 sigma 值
    def get_adjacent_sigma(self, timesteps, t):
        # 如果当前时间步为 0，则返回与 t 相同形状的零张量，否则返回离散 sigma 值
        return torch.where(
            timesteps == 0,
            torch.zeros_like(t.to(timesteps.device)),
            self.discrete_sigmas[timesteps - 1].to(timesteps.device),
        )

    # 步骤预测，处理模型输出、当前时间步和样本
    def step_pred(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    # 步骤校正，处理模型输出和样本
    def step_correct(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        根据网络的 `model_output` 校正预测样本。通常在对前一个时间步的预测后重复运行此函数。

        参数：
            model_output (`torch.Tensor`):
                来自学习扩散模型的直接输出。
            sample (`torch.Tensor`):
                通过扩散过程创建的当前样本实例。
            generator (`torch.Generator`, *可选*):
                随机数生成器。
            return_dict (`bool`, *可选*, 默认值为 `True`):
                是否返回 [`~schedulers.scheduling_sde_ve.SdeVeOutput`] 或 `tuple`。

        返回：
            [`~schedulers.scheduling_sde_ve.SdeVeOutput`] 或 `tuple`:
                如果 `return_dict` 为 `True`，返回 [`~schedulers.scheduling_sde_ve.SdeVeOutput`]，否则返回一个元组
                其第一个元素是样本张量。

        """
        # 检查时间步是否被设置，如果没有则抛出错误
        if self.timesteps is None:
            raise ValueError(
                "`self.timesteps` 未设置，需要在创建调度器后运行 'set_timesteps'"
            )

        # 对于小批量大小，论文建议用 sqrt(d) 替代 norm(z)，其中 d 是 z 的维度
        # 为校正生成噪声
        noise = randn_tensor(sample.shape, layout=sample.layout, generator=generator).to(sample.device)

        # 从 model_output、噪声和信噪比计算步长
        grad_norm = torch.norm(model_output.reshape(model_output.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (self.config.snr * noise_norm / grad_norm) ** 2 * 2
        step_size = step_size * torch.ones(sample.shape[0]).to(sample.device)
        # self.repeat_scalar(step_size, sample.shape[0])

        # 计算校正样本：model_output 项和噪声项
        step_size = step_size.flatten()
        while len(step_size.shape) < len(sample.shape):
            step_size = step_size.unsqueeze(-1)  # 扩展 step_size 以匹配样本的维度
        prev_sample_mean = sample + step_size * model_output  # 计算样本的平均值
        prev_sample = prev_sample_mean + ((step_size * 2) ** 0.5) * noise  # 加上噪声以得到最终样本

        if not return_dict:  # 如果不返回字典，则以元组形式返回
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)  # 返回校正后的样本
    # 返回一个张量，表示加噪样本
    ) -> torch.Tensor:
        # 确保时间步和 sigma 与原始样本具有相同的设备和数据类型
        timesteps = timesteps.to(original_samples.device)
        # 获取与时间步对应的 sigma 值，并确保它们在相同的设备上
        sigmas = self.discrete_sigmas.to(original_samples.device)[timesteps]
        # 如果存在噪声，则将其按 sigma 进行缩放；否则生成与原始样本相同形状的随机噪声
        noise = (
            noise * sigmas[:, None, None, None]
            if noise is not None
            else torch.randn_like(original_samples) * sigmas[:, None, None, None]
        )
        # 将噪声添加到原始样本，生成加噪样本
        noisy_samples = noise + original_samples
        # 返回加噪样本
        return noisy_samples
    
    # 定义返回训练时间步数的函数
    def __len__(self):
        # 返回配置中的训练时间步数
        return self.config.num_train_timesteps
```