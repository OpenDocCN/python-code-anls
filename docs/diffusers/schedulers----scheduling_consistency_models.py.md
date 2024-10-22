# `.\diffusers\schedulers\scheduling_consistency_models.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）进行许可；
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件在许可证下分发是以“原样”基础进行的，
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 请参见许可证以获取管理权限和
# 限制的具体语言。

from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from typing import List, Optional, Tuple, Union  # 导入类型注解

import numpy as np  # 导入 numpy 库并简写为 np
import torch  # 导入 PyTorch 库

from ..configuration_utils import ConfigMixin, register_to_config  # 从上级模块导入配置相关的工具
from ..utils import BaseOutput, logging  # 从上级模块导入基本输出类和日志工具
from ..utils.torch_utils import randn_tensor  # 从上级模块导入随机张量生成函数
from .scheduling_utils import SchedulerMixin  # 从当前模块导入调度器混合类

logger = logging.get_logger(__name__)  # 创建一个日志记录器，使用当前模块的名称，禁用 pylint 命名检查

@dataclass
class CMStochasticIterativeSchedulerOutput(BaseOutput):  # 定义一个数据类，继承自 BaseOutput
    """
    调度器的 `step` 函数的输出类。

    参数：
        prev_sample (`torch.Tensor`，形状为 `(batch_size, num_channels, height, width)`，用于图像):
            计算的前一步样本 `(x_{t-1})`。`prev_sample` 应作为下一个模型输入用于
            去噪循环。
    """

    prev_sample: torch.Tensor  # 前一步样本的张量属性

class CMStochasticIterativeScheduler(SchedulerMixin, ConfigMixin):  # 定义调度器类，继承自 SchedulerMixin 和 ConfigMixin
    """
    一致性模型的多步和一步采样。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查看超类文档以了解库为所有调度器实现的
    通用方法，例如加载和保存。
    # 函数参数说明
    Args:
        num_train_timesteps (`int`, defaults to 40):  # 训练模型的扩散步骤数，默认为40
            The number of diffusion steps to train the model.
        sigma_min (`float`, defaults to 0.002):  # sigma调度中的最小噪声幅度，默认为0.002
            Minimum noise magnitude in the sigma schedule. Defaults to 0.002 from the original implementation.
        sigma_max (`float`, defaults to 80.0):  # sigma调度中的最大噪声幅度，默认为80.0
            Maximum noise magnitude in the sigma schedule. Defaults to 80.0 from the original implementation.
        sigma_data (`float`, defaults to 0.5):  # EDM中数据分布的标准差，默认为0.5
            The standard deviation of the data distribution from the EDM
            [paper](https://huggingface.co/papers/2206.00364). Defaults to 0.5 from the original implementation.
        s_noise (`float`, defaults to 1.0):  # 用于采样时抵消细节丢失的额外噪声量，默认为1.0
            The amount of additional noise to counteract loss of detail during sampling. A reasonable range is [1.000,
            1.011]. Defaults to 1.0 from the original implementation.
        rho (`float`, defaults to 7.0):  # 用于从EDM计算Karras sigma调度的参数，默认为7.0
            The parameter for calculating the Karras sigma schedule from the EDM
            [paper](https://huggingface.co/papers/2206.00364). Defaults to 7.0 from the original implementation.
        clip_denoised (`bool`, defaults to `True`):  # 是否将去噪输出限制在`(-1, 1)`范围内
            Whether to clip the denoised outputs to `(-1, 1)`.
        timesteps (`List` or `np.ndarray` or `torch.Tensor`, *optional*):  # 可选的明确时间步调度，需按升序排列
            An explicit timestep schedule that can be optionally specified. The timesteps are expected to be in
            increasing order.
    """

    # 设置步骤的初始顺序为1
    order = 1

    @register_to_config
    def __init__((
        self,
        num_train_timesteps: int = 40,  # 设置训练时的扩散步骤数
        sigma_min: float = 0.002,  # 设置sigma调度的最小噪声幅度
        sigma_max: float = 80.0,  # 设置sigma调度的最大噪声幅度
        sigma_data: float = 0.5,  # 设置数据分布的标准差
        s_noise: float = 1.0,  # 设置额外噪声量
        rho: float = 7.0,  # 设置Karras sigma调度的参数
        clip_denoised: bool = True,  # 设置是否对去噪输出进行限制
    ):
        # 初始化噪声分布的标准差
        self.init_noise_sigma = sigma_max

        # 创建从0到1的线性渐变，长度为训练步骤数
        ramp = np.linspace(0, 1, num_train_timesteps)
        # 将渐变转换为Karras sigma
        sigmas = self._convert_to_karras(ramp)
        # 将sigma转换为时间步
        timesteps = self.sigma_to_t(sigmas)

        # 可设值初始化
        self.num_inference_steps = None  # 推理步骤数初始化为None
        # 将sigma转换为torch张量
        self.sigmas = torch.from_numpy(sigmas)
        # 将时间步转换为torch张量
        self.timesteps = torch.from_numpy(timesteps)
        self.custom_timesteps = False  # 设置自定义时间步标志为False
        self.is_scale_input_called = False  # 标记是否调用了输入缩放
        self._step_index = None  # 初始化步骤索引为None
        self._begin_index = None  # 初始化开始索引为None
        # 将sigma移动到CPU，避免过多的CPU/GPU通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加1。
        """
        return self._step_index  # 返回当前步骤索引

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应该通过`set_begin_index`方法从管道中设置。
        """
        return self._begin_index  # 返回开始索引

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制的代码
    # 设置调度器的起始索引，默认为 0
    def set_begin_index(self, begin_index: int = 0):
        # 调用此函数应在推理前执行
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.
    
        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        # 将给定的起始索引存储到实例变量中
        self._begin_index = begin_index
    
    # 缩放一致性模型输入，通过公式 `(sigma**2 + sigma_data**2) ** 0.5`
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        # 获取与当前时间步对应的 sigma 值
        """
        Scales the consistency model input by `(sigma**2 + sigma_data**2) ** 0.5`.
    
        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`float` or `torch.Tensor`):
                The current timestep in the diffusion chain.
    
        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        # 如果步骤索引为 None，则初始化步骤索引
        if self.step_index is None:
            self._init_step_index(timestep)
    
        # 从 sigmas 列表中获取当前步骤的 sigma 值
        sigma = self.sigmas[self.step_index]
    
        # 按照给定公式缩放输入样本
        sample = sample / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
    
        # 标记已调用缩放输入的方法
        self.is_scale_input_called = True
        # 返回缩放后的样本
        return sample
    
    # 从 Karras sigmas 获取缩放时间步，以输入一致性模型
    def sigma_to_t(self, sigmas: Union[float, np.ndarray]):
        # 
        """
        Gets scaled timesteps from the Karras sigmas for input to the consistency model.
    
        Args:
            sigmas (`float` or `np.ndarray`):
                A single Karras sigma or an array of Karras sigmas.
    
        Returns:
            `float` or `np.ndarray`:
                A scaled input timestep or scaled input timestep array.
        """
        # 检查 sigmas 是否为 numpy 数组，如果不是，则转换为数组
        if not isinstance(sigmas, np.ndarray):
            sigmas = np.array(sigmas, dtype=np.float64)
    
        # 根据 Karras 的公式计算时间步
        timesteps = 1000 * 0.25 * np.log(sigmas + 1e-44)
    
        # 返回计算得到的时间步
        return timesteps
    
    # 设置推理的时间步，参数为可选
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    # 修改后的 _convert_to_karras 实现，接受 ramp 作为参数
    def _convert_to_karras(self, ramp):
        # 构造 Karras 等人的噪声调度
        """Constructs the noise schedule of Karras et al. (2022)."""
    
        # 获取配置中的最小和最大 sigma 值
        sigma_min: float = self.config.sigma_min
        sigma_max: float = self.config.sigma_max
    
        # 获取配置中的 rho 值
        rho = self.config.rho
        # 计算最小和最大 sigma 的倒数
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 根据 ramp 计算 sigma 值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回计算得到的 sigma 值
        return sigmas
    
    # 获取缩放因子
    def get_scalings(self, sigma):
        # 从配置中获取 sigma_data 值
        sigma_data = self.config.sigma_data
    
        # 计算 c_skip 和 c_out
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        # 返回计算得到的缩放因子
        return c_skip, c_out
    # 获取用于一致性模型参数化中的缩放因子，以满足边界条件
    def get_scalings_for_boundary_condition(self, sigma):
        """
        获取用于一致性模型参数化的缩放因子（参见论文的附录 C），以强制边界条件。

        <Tip>

        在 `c_skip` 和 `c_out` 的方程中，`epsilon` 被设置为 `sigma_min`。

        </Tip>

        参数:
            sigma (`torch.Tensor`):
                当前的 sigma 值，来自 Karras sigma 调度。

        返回:
            `tuple`:
                一个包含两个元素的元组，其中 `c_skip`（当前样本的权重）是第一个元素，`c_out`
                （一致性模型输出的权重）是第二个元素。
        """
        # 从配置中获取最小的 sigma 值
        sigma_min = self.config.sigma_min
        # 从配置中获取数据 sigma 值
        sigma_data = self.config.sigma_data

        # 计算 c_skip 的缩放因子
        c_skip = sigma_data**2 / ((sigma - sigma_min) ** 2 + sigma_data**2)
        # 计算 c_out 的缩放因子
        c_out = (sigma - sigma_min) * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        # 返回 c_skip 和 c_out
        return c_skip, c_out

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep 复制
    # 获取给定时间步的索引
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果未提供调度时间步，则使用默认时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与当前时间步匹配的索引
        indices = (schedule_timesteps == timestep).nonzero()

        # 对于**第一个** `step` 采取的 sigma 索引
        # 始终是第二个索引（或如果只有一个，则为最后一个索引）
        # 这样可以确保在去噪调度中间开始时不会意外跳过一个 sigma
        pos = 1 if len(indices) > 1 else 0

        # 返回对应索引的值
        return indices[pos].item()

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制
    # 初始化步骤索引
    def _init_step_index(self, timestep):
        # 如果开始索引为 None，计算当前时间步的索引
        if self.begin_index is None:
            # 如果时间步是张量，将其转换为与时间步设备相同
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 根据时间步获取步骤索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则，使用已定义的开始索引
            self._step_index = self._begin_index

    # 执行一步操作
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise 复制
    # 向样本添加噪声
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    # 函数返回一个张量，表示处理后的噪声样本
    ) -> torch.Tensor:
        # 将 sigmas 转移到与原始样本相同的设备和数据类型
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备类型是否为 mps，且 timesteps 是否为浮点型
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            # 将 timesteps 转移到相同设备，转换为 float32
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将 timesteps 转移到原始样本的设备，不改变数据类型
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)
    
        # 当 scheduler 用于训练或管道未实现 set_begin_index 时，begin_index 为 None
        if self.begin_index is None:
            # 根据每个 timesteps 找到对应的步索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一次去噪步骤后调用 add_noise（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一次去噪步骤前调用 add_noise 创建初始潜在图像 (img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]
    
        # 根据步索引获取对应的 sigmas，并展平为一维
        sigma = sigmas[step_indices].flatten()
        # 如果 sigma 的维度小于原始样本的维度，则在最后添加一个维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
    
        # 将噪声样本与 sigma 相加，生成带噪声的样本
        noisy_samples = original_samples + noise * sigma
        # 返回带噪声的样本
        return noisy_samples
    
    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```