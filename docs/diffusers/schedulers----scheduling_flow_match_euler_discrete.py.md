# `.\diffusers\schedulers\scheduling_flow_match_euler_discrete.py`

```py
# 版权声明，说明文件归 Stability AI、Katherine Crowson 和 HuggingFace 团队所有
# 
# 根据 Apache 许可证 2.0 版进行许可，用户必须遵守该许可证才能使用本文件
# 用户可在以下网址获取许可证的副本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律或书面协议另有约定，软件以“原样”方式分发，不提供任何形式的明示或暗示的担保或条件
# 具体权限和限制请参阅许可证
import math  # 导入数学模块以进行数学计算
from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import List, Optional, Tuple, Union  # 从 typing 模块导入类型注解

import numpy as np  # 导入 numpy 库以进行数组和矩阵操作
import torch  # 导入 PyTorch 库以进行深度学习操作

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入混合类和注册函数
from ..utils import BaseOutput, logging  # 从工具模块导入基础输出类和日志记录工具
from .scheduling_utils import SchedulerMixin  # 从调度工具导入调度混合类


logger = logging.get_logger(__name__)  # 创建一个日志记录器，用于记录本模块的日志信息


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.FloatTensor`，形状为 `(batch_size, num_channels, height, width)` 的图像):
            计算的上一个时间步的样本 `(x_{t-1})`。 `prev_sample` 应作为下一次模型输入在去噪循环中使用。
    """

    prev_sample: torch.FloatTensor  # 上一个时间步的样本，使用 PyTorch 的浮点张量


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    欧拉调度器。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关库为所有调度器实现的通用方法的文档，请查看父类文档。

    参数：
        num_train_timesteps (`int`，默认为 1000):
            用于训练模型的扩散步骤数量。
        timestep_spacing (`str`，默认为 `"linspace"`):
            时间步的缩放方式。有关更多信息，请参阅 [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) 表 2。
        shift (`float`，默认为 1.0):
            时间步调度的偏移值。
    """

    _compatibles = []  # 兼容性列表，初始化为空
    order = 1  # 调度器的顺序设置为 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,  # 设置训练时间步数量，默认值为 1000
        shift: float = 1.0,  # 设置时间步偏移值，默认值为 1.0
        use_dynamic_shifting=False,  # 是否使用动态偏移，默认值为 False
        base_shift: Optional[float] = 0.5,  # 基础偏移值，默认值为 0.5
        max_shift: Optional[float] = 1.15,  # 最大偏移值，默认值为 1.15
        base_image_seq_len: Optional[int] = 256,  # 基础图像序列长度，默认值为 256
        max_image_seq_len: Optional[int] = 4096,  # 最大图像序列长度，默认值为 4096
    ):
        # 创建一个从1到num_train_timesteps的等间隔数组，并反转其顺序
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        # 将NumPy数组转换为PyTorch张量，并指定数据类型为float32
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        # 计算sigmas，作为timesteps与num_train_timesteps的比例
        sigmas = timesteps / num_train_timesteps
        if not use_dynamic_shifting:
            # 如果不使用动态偏移，则根据shift调整sigmas的值
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        # 计算最终的timesteps，乘以num_train_timesteps以获得真实的时间步长
        self.timesteps = sigmas * num_train_timesteps

        # 初始化步骤索引和开始索引为None
        self._step_index = None
        self._begin_index = None

        # 将sigmas移动到CPU，以减少CPU/GPU之间的通信
        self.sigmas = sigmas.to("cpu")  
        # 获取sigmas中的最小值
        self.sigma_min = self.sigmas[-1].item()
        # 获取sigmas中的最大值
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度步骤后增加1。
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应该通过`set_begin_index`方法从管道设置。
        """
        return self._begin_index

    # 从diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index复制而来
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的开始索引。该函数应在推理之前从管道运行。

        Args:
            begin_index (`int`):
                调度器的开始索引。
        """
        # 设置开始索引
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        前向过程在流匹配中

        参数:
            sample (`torch.FloatTensor`):
                输入样本。
            timestep (`int`, *可选*):
                当前扩散链中的时间步。

        返回:
            `torch.FloatTensor`:
                缩放后的输入样本。
        """
        # 确保 sigmas 和 timesteps 的设备和数据类型与原始样本相同
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        # 检查设备类型是否为 MPS，且时间步是否为浮点数
        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            # MPS 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)  # 将时间步转换为 float32
            timestep = timestep.to(sample.device, dtype=torch.float32)  # 将时间步转换为 float32
        else:
            # 将时间步转换到样本的设备上，保持原始数据类型
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)  # 将时间步转移到样本的设备上

        # 当 scheduler 用于训练时，self.begin_index 为 None，或 pipeline 没有实现 set_begin_index
        if self.begin_index is None:
            # 根据当前时间步索引计算步骤索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # 在第一次去噪步骤后调用 add_noise（用于修补）
            step_indices = [self.step_index] * timestep.shape[0]  # 重复当前步骤索引
        else:
            # 在第一次去噪步骤之前调用 add_noise 以创建初始潜变量（图像到图像）
            step_indices = [self.begin_index] * timestep.shape[0]  # 重复初始步骤索引

        # 获取对应步骤的 sigma 值并展平
        sigma = sigmas[step_indices].flatten()
        # 扩展 sigma 的维度以匹配样本的形状
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        # 通过 sigma 和噪声对样本进行加权组合
        sample = sigma * noise + (1.0 - sigma) * sample

        # 返回处理后的样本
        return sample

    # 将 sigma 转换为时间步 t
    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps  # 计算对应的时间步

    # 根据给定的 mu、sigma 和时间张量 t 进行时间偏移计算
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)  # 计算时间偏移的值

    # 设置时间步的函数
    def set_timesteps(
        self,
        num_inference_steps: int = None,  # 推理步骤数量（可选）
        device: Union[str, torch.device] = None,  # 设备类型（可选）
        sigmas: Optional[List[float]] = None,  # sigma 值列表（可选）
        mu: Optional[float] = None,  # mu 值（可选）
    ):
        """
        设置用于扩散链的离散时间步（在推理前运行）。

        参数：
            num_inference_steps (`int`):
                用于生成样本的扩散步骤数量，使用预训练模型时。
            device (`str` 或 `torch.device`, *可选*):
                时间步要移动到的设备。如果为 `None`，则时间步不被移动。
        """

        # 如果使用动态移动并且 mu 为 None，则抛出值错误
        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        # 如果 sigmas 为 None，则初始化 num_inference_steps 和计算时间步
        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            # 生成从 sigma_max 到 sigma_min 的均匀时间步
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            # 归一化时间步以计算 sigmas
            sigmas = timesteps / self.config.num_train_timesteps

        # 如果使用动态移动，调用时间移动函数处理 sigmas
        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            # 使用配置中的偏移量调整 sigmas
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        # 将 sigmas 转换为 PyTorch 张量，并设置数据类型和设备
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        # 计算时间步与训练时间步的乘积
        timesteps = sigmas * self.config.num_train_timesteps

        # 将计算出的时间步移动到指定设备
        self.timesteps = timesteps.to(device=device)
        # 将 sigmas 与一个零张量拼接，扩展维度
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        # 初始化步骤索引和开始索引为 None
        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果没有提供计划时间步，则使用当前的时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与给定时间步相等的索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 对于第一次“步骤”，所采用的 sigma 索引始终是第二个索引
        # 如果只有一个索引，则使用最后一个索引
        pos = 1 if len(indices) > 1 else 0

        # 返回所需索引的值
        return indices[pos].item()

    def _init_step_index(self, timestep):
        # 如果开始索引为 None，则计算步骤索引
        if self.begin_index is None:
            # 如果时间步是张量，移动到当前时间步设备
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 通过时间步索引初始化步骤索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则，使用开始索引初始化步骤索引
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,  # 模型输出的浮点张量
        timestep: Union[float, torch.FloatTensor],  # 当前的时间步，可以是浮点数或张量
        sample: torch.FloatTensor,  # 当前样本的浮点张量
        s_churn: float = 0.0,  # 额外的旋涡参数，默认值为 0.0
        s_tmin: float = 0.0,  # 最小时间步限制，默认值为 0.0
        s_tmax: float = float("inf"),  # 最大时间步限制，默认值为无穷大
        s_noise: float = 1.0,  # 噪声强度，默认值为 1.0
        generator: Optional[torch.Generator] = None,  # 随机数生成器，默认值为 None
        return_dict: bool = True,  # 是否返回字典格式的结果，默认值为 True
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        通过反向 SDE 从上一个时间步预测样本。该函数从学习模型输出（通常是预测的噪声）传播扩散过程。

        参数：
            model_output (`torch.FloatTensor`):
                来自学习扩散模型的直接输出。
            timestep (`float`):
                当前扩散链中的离散时间步。
            sample (`torch.FloatTensor`):
                当前通过扩散过程创建的样本实例。
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                添加到样本的噪声缩放因子。
            generator (`torch.Generator`, *optional*):
                随机数生成器。
            return_dict (`bool`):
                是否返回 [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] 或
                元组。

        返回：
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`]，
                否则返回一个元组，元组的第一个元素是样本张量。
        """

        # 检查 timestep 是否为整数类型
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            # 如果 timestep 是整数类型，抛出异常，提示不支持此用法
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        # 如果 step_index 为空，初始化 step_index
        if self.step_index is None:
            self._init_step_index(timestep)

        # 为了避免计算 prev_sample 时的精度问题，将样本上升为 float32 类型
        sample = sample.to(torch.float32)

        # 获取当前和下一个时间步的 sigma 值
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        # 计算上一个样本，依据当前样本和模型输出
        prev_sample = sample + (sigma_next - sigma) * model_output

        # 将样本转换回与模型兼容的数据类型
        prev_sample = prev_sample.to(model_output.dtype)

        # 完成后将步骤索引增加一
        self._step_index += 1

        # 如果不需要返回字典，则返回包含 prev_sample 的元组
        if not return_dict:
            return (prev_sample,)

        # 返回 FlowMatchEulerDiscreteSchedulerOutput 对象，包含 prev_sample
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```