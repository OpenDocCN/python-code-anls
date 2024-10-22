# `.\diffusers\schedulers\scheduling_edm_euler.py`

```py
# 版权信息，声明文件所有权和使用许可
# Copyright 2024 Katherine Crowson and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 (“许可证”) 授权使用
# 你只能在遵守许可证的情况下使用此文件
# 你可以在以下网址获取许可证副本
# 
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件按“现状”分发
# 不提供任何形式的担保或条件
# 有关许可的特定条款和限制，请参阅许可证

import math  # 导入数学库以进行数学运算
from dataclasses import dataclass  # 从dataclasses导入dataclass以简化类定义
from typing import Optional, Tuple, Union  # 从typing导入类型注解支持

import torch  # 导入PyTorch库以进行张量计算

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入配置混合类和注册功能
from ..utils import BaseOutput, logging  # 从工具库导入基础输出类和日志功能
from ..utils.torch_utils import randn_tensor  # 从torch工具导入随机张量生成函数
from .scheduling_utils import SchedulerMixin  # 从调度工具导入调度混合类

logger = logging.get_logger(__name__)  # 创建当前模块的日志记录器，方便调试和记录信息

@dataclass
# 定义调度器的输出类，继承自BaseOutput，提供先前样本和预测样本
class EDMEulerSchedulerOutput(BaseOutput):
    """
    调度器 `step` 函数输出的输出类。

    参数:
        prev_sample (`torch.Tensor` 的形状为 `(batch_size, num_channels, height, width)`，用于图像):
            计算的先前时间步样本 `(x_{t-1})`。`prev_sample` 应作为下一个模型输入使用
            在去噪循环中。
        pred_original_sample (`torch.Tensor` 的形状为 `(batch_size, num_channels, height, width)`，用于图像):
            基于当前时间步模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或指导。
    """

    prev_sample: torch.Tensor  # 定义先前样本的属性，类型为张量
    pred_original_sample: Optional[torch.Tensor] = None  # 定义可选的预测原始样本属性，类型为张量


class EDMEulerScheduler(SchedulerMixin, ConfigMixin):
    """
    实现Karras等人2022年提出的EDM公式中的Euler调度器 [1]。

    [1] Karras, Tero, 等。“阐明基于扩散的生成模型的设计空间。”
    https://arxiv.org/abs/2206.00364

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关所有调度程序实现的通用
    方法的文档，请查看父类文档，例如加载和保存。
    # 参数说明文档
    Args:
        sigma_min (`float`, *optional*, defaults to 0.002):
            # sigma 调度中的最小噪声幅度，EDM 论文中设置为 0.002；合理范围是 [0, 10]。
            Minimum noise magnitude in the sigma schedule. This was set to 0.002 in the EDM paper [1]; a reasonable
            range is [0, 10].
        sigma_max (`float`, *optional*, defaults to 80.0):
            # sigma 调度中的最大噪声幅度，EDM 论文中设置为 80.0；合理范围是 [0.2, 80.0]。
            Maximum noise magnitude in the sigma schedule. This was set to 80.0 in the EDM paper [1]; a reasonable
            range is [0.2, 80.0].
        sigma_data (`float`, *optional*, defaults to 0.5):
            # 数据分布的标准差，EDM 论文中设置为 0.5。
            The standard deviation of the data distribution. This is set to 0.5 in the EDM paper [1].
        sigma_schedule (`str`, *optional*, defaults to `karras`):
            # 用于计算 `sigmas` 的 sigma 调度，默认使用 EDM 论文中介绍的调度。
            Sigma schedule to compute the `sigmas`. By default, we the schedule introduced in the EDM paper
            (https://arxiv.org/abs/2206.00364). Other acceptable value is "exponential". The exponential schedule was
            incorporated in this model: https://huggingface.co/stabilityai/cosxl.
        num_train_timesteps (`int`, defaults to 1000):
            # 训练模型的扩散步骤数量。
            The number of diffusion steps to train the model.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            # 调度函数的预测类型，可以是 `epsilon`、`sample` 或 `v_prediction`。
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        rho (`float`, *optional*, defaults to 7.0):
            # 用于计算 Karras sigma 调度的 rho 参数，EDM 论文中设置为 7.0。
            The rho parameter used for calculating the Karras sigma schedule, which is set to 7.0 in the EDM paper [1].
    """

    # 初始化一个空列表，用于存储兼容的设置
    _compatibles = []
    # 设置某个顺序标识，默认为 1
    order = 1

    @register_to_config
    # 初始化函数，用于设置对象的基本属性
    def __init__(
        self,
        sigma_min: float = 0.002,  # 设置最小噪声幅度，默认值为 0.002
        sigma_max: float = 80.0,    # 设置最大噪声幅度，默认值为 80.0
        sigma_data: float = 0.5,    # 设置数据分布的标准差，默认值为 0.5
        sigma_schedule: str = "karras",  # 设置 sigma 调度，默认值为 "karras"
        num_train_timesteps: int = 1000,  # 设置训练时的扩散步骤数，默认值为 1000
        prediction_type: str = "epsilon",  # 设置预测类型，默认值为 "epsilon"
        rho: float = 7.0,  # 设置 rho 参数，默认值为 7.0
    ):
        # 验证 sigma 调度类型是否有效，若无效则抛出异常
        if sigma_schedule not in ["karras", "exponential"]:
            raise ValueError(f"Wrong value for provided for `{sigma_schedule=}`.`")

        # 可设置的值，初始化为空
        self.num_inference_steps = None

        # 创建一个线性 ramp 从 0 到 1，包含 num_train_timesteps 个点
        ramp = torch.linspace(0, 1, num_train_timesteps)
        # 如果选择的调度是 "karras"，计算对应的 sigmas
        if sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(ramp)
        # 如果选择的调度是 "exponential"，计算对应的 sigmas
        elif sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(ramp)

        # 预处理噪声，获得时间步信息
        self.timesteps = self.precondition_noise(sigmas)

        # 将 sigmas 和一个零数组合并，后者位于计算设备上
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        # 标记是否已调用输入缩放
        self.is_scale_input_called = False

        # 初始化步骤索引和开始索引为空
        self._step_index = None
        self._begin_index = None
        # 将 sigmas 移动到 CPU，避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    # 获取初始噪声分布的标准差
    def init_noise_sigma(self):
        # 计算初始噪声分布的标准差
        return (self.config.sigma_max**2 + 1) ** 0.5

    @property
    # 定义当前时间步的索引计数器，每次调度器步骤后增加1
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加1。
        """
        # 返回当前步骤索引
        return self._step_index

    # 定义一个属性，表示开始索引
    @property
    def begin_index(self):
        """
        第一个时间步的索引。应该通过 `set_begin_index` 方法从管道设置。
        """
        # 返回开始索引
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制而来
    # 设置开始索引的方法
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的开始索引。此函数应在推理之前从管道运行。

        Args:
            begin_index (`int`):
                调度器的开始索引。
        """
        # 将开始索引设置为给定值
        self._begin_index = begin_index

    # 预处理输入样本的方法
    def precondition_inputs(self, sample, sigma):
        # 计算输入样本的缩放因子
        c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
        # 根据缩放因子调整样本
        scaled_sample = sample * c_in
        # 返回缩放后的样本
        return scaled_sample

    # 预处理噪声的方法
    def precondition_noise(self, sigma):
        # 检查 sigma 是否为张量，如果不是则转换为张量
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma])
        
        # 计算噪声的缩放因子
        c_noise = 0.25 * torch.log(sigma)

        # 返回噪声的缩放因子
        return c_noise

    # 预处理输出的方法
    def precondition_outputs(self, sample, model_output, sigma):
        # 获取配置中的 sigma_data
        sigma_data = self.config.sigma_data
        # 计算跳过的系数
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

        # 根据预测类型计算输出的系数
        if self.config.prediction_type == "epsilon":
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        elif self.config.prediction_type == "v_prediction":
            c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        else:
            # 如果预测类型不支持，则抛出异常
            raise ValueError(f"Prediction type {self.config.prediction_type} is not supported.")

        # 计算去噪后的结果
        denoised = c_skip * sample + c_out * model_output

        # 返回去噪后的结果
        return denoised

    # 缩放模型输入的方法
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器互换性。通过 `(sigma**2 + 1) ** 0.5` 缩放去噪模型输入，以匹配欧拉算法。

        Args:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *optional*):
                扩散链中的当前时间步。

        Returns:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 检查当前步骤索引是否为 None，如果是则初始化步骤索引
        if self.step_index is None:
            self._init_step_index(timestep)

        # 获取当前步骤对应的 sigma 值
        sigma = self.sigmas[self.step_index]
        # 预处理输入样本
        sample = self.precondition_inputs(sample, sigma)

        # 标记输入缩放方法已被调用
        self.is_scale_input_called = True
        # 返回预处理后的样本
        return sample
    # 设置离散时间步，用于扩散链（推理前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步（在推理前运行）。

        参数：
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数。
            device (`str` 或 `torch.device`, *可选*):
                时间步应该移动到的设备。如果为 `None`，则不移动时间步。
        """
        # 将输入的扩散步骤数存储在实例变量中
        self.num_inference_steps = num_inference_steps

        # 创建一个从0到1的线性空间，包含num_inference_steps个点
        ramp = torch.linspace(0, 1, self.num_inference_steps)
        # 根据配置选择Karras噪声调度
        if self.config.sigma_schedule == "karras":
            # 计算Karras噪声调度的sigma值
            sigmas = self._compute_karras_sigmas(ramp)
        # 根据配置选择指数噪声调度
        elif self.config.sigma_schedule == "exponential":
            # 计算指数噪声调度的sigma值
            sigmas = self._compute_exponential_sigmas(ramp)

        # 将sigmas转换为float32类型，并移动到指定设备
        sigmas = sigmas.to(dtype=torch.float32, device=device)
        # 预处理噪声并存储在timesteps中
        self.timesteps = self.precondition_noise(sigmas)

        # 将sigma与零向量连接，以便扩散过程中使用
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        # 初始化步索引和开始索引为None
        self._step_index = None
        self._begin_index = None
        # 将sigmas移动到CPU以避免过多的CPU/GPU通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    # 从Karras等人（2022）构建噪声调度
    def _compute_karras_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        """构建Karras等人（2022）的噪声调度。"""
        # 如果未提供sigma_min，则使用配置中的值
        sigma_min = sigma_min or self.config.sigma_min
        # 如果未提供sigma_max，则使用配置中的值
        sigma_max = sigma_max or self.config.sigma_max

        # 从配置中获取rho值
        rho = self.config.rho
        # 计算sigma_min和sigma_max的倒数
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 计算根据ramp的sigma值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # 计算指数噪声调度的sigma值
    def _compute_exponential_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        """实现紧随k-diffusion。"""
        # 如果未提供sigma_min，则使用配置中的值
        sigma_min = sigma_min or self.config.sigma_min
        # 如果未提供sigma_max，则使用配置中的值
        sigma_max = sigma_max or self.config.sigma_max
        # 计算从sigma_min到sigma_max的对数线性空间，并取指数后翻转
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), len(ramp)).exp().flip(0)
        return sigmas

    # 从diffusers库复制的调度器的时间步索引计算方法
    # 根据时间步计算索引
        def index_for_timestep(self, timestep, schedule_timesteps=None):
            # 如果没有提供调度时间步，使用实例的时间步
            if schedule_timesteps is None:
                schedule_timesteps = self.timesteps
    
            # 找出与给定时间步相等的时间步的索引
            indices = (schedule_timesteps == timestep).nonzero()
    
            # 确定要使用的 sigma 索引，默认为第二个索引
            # 如果只有一个索引，则使用第一个索引
            pos = 1 if len(indices) > 1 else 0
    
            # 返回选定索引的标量值
            return indices[pos].item()
    
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index 复制
        def _init_step_index(self, timestep):
            # 如果开始索引为 None，初始化步索引
            if self.begin_index is None:
                # 如果时间步是张量，则将其移动到相应设备
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.to(self.timesteps.device)
                # 计算时间步对应的步索引
                self._step_index = self.index_for_timestep(timestep)
            else:
                # 否则直接使用开始索引
                self._step_index = self._begin_index
    
        # 定义步进函数，包含多个参数
        def step(
            self,
            model_output: torch.Tensor,
            timestep: Union[float, torch.Tensor],
            sample: torch.Tensor,
            s_churn: float = 0.0,
            s_tmin: float = 0.0,
            s_tmax: float = float("inf"),
            s_noise: float = 1.0,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise 复制
        def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    # 返回一个张量，表示噪声样本
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 与 original_samples 拥有相同的设备和数据类型
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备类型为 mps 且 timesteps 是浮点类型
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将 timesteps 转换为 original_samples 设备的数据类型
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)
    
        # self.begin_index 为 None 表示调度器用于训练，或管道未实现 set_begin_index
        if self.begin_index is None:
            # 通过调度时间步计算步索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一次去噪步骤后调用 add_noise（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一次去噪步骤前调用 add_noise 创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]
    
        # 根据步索引获取 sigmas，并扁平化
        sigma = sigmas[step_indices].flatten()
        # 扩展 sigma 的维度以匹配 original_samples 的维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
    
        # 生成噪声样本，原始样本加上噪声乘以 sigma
        noisy_samples = original_samples + noise * sigma
        # 返回噪声样本
        return noisy_samples
    
    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```