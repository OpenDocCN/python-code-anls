# `.\diffusers\schedulers\scheduling_edm_dpmsolver_multistep.py`

```py
# 版权所有 2024 TSAIL Team 和 The HuggingFace Team。保留所有权利。
#
# 根据 Apache License, Version 2.0（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，按“原样”分发软件，
# 不提供任何形式的保证或条件，无论是明示或暗示。
# 有关许可证所涵盖的特定权限和限制，请参见许可证。

# 免责声明：此文件受到 https://github.com/LuChengTHU/dpm-solver 和 https://github.com/NVlabs/edm 的强烈影响

# 导入数学库
import math
# 从 typing 模块导入必要的类型提示
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库，通常用于数组和数学运算
import numpy as np
# 导入 PyTorch 库，用于深度学习操作
import torch

# 从配置工具中导入 ConfigMixin 和 register_to_config
from ..configuration_utils import ConfigMixin, register_to_config
# 从 Torch 工具中导入随机张量生成函数
from ..utils.torch_utils import randn_tensor
# 从调度工具中导入调度器混合类和调度器输出类
from .scheduling_utils import SchedulerMixin, SchedulerOutput


class EDMDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    实现 Karras 等人 2022 年提出的 EDM 形式的 DPMSolverMultistepScheduler [1]。
    `EDMDPMSolverMultistepScheduler` 是用于扩散 ODE 的快速专用高阶求解器。

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关所有调度器的通用方法的文档，
    请查看超类文档，例如加载和保存。

    """

    # 定义兼容性列表，初始化为空
    _compatibles = []
    # 定义求解器的阶数，默认为 1
    order = 1

    @register_to_config
    # 初始化函数，接收多个参数，具有默认值
    def __init__(
        # 最小 sigma 值，默认 0.002
        self,
        sigma_min: float = 0.002,
        # 最大 sigma 值，默认 80.0
        sigma_max: float = 80.0,
        # 数据的 sigma 值，默认 0.5
        sigma_data: float = 0.5,
        # sigma 的调度类型，默认为 "karras"
        sigma_schedule: str = "karras",
        # 训练的时间步数，默认 1000
        num_train_timesteps: int = 1000,
        # 预测类型，默认为 "epsilon"
        prediction_type: str = "epsilon",
        # ρ 值，默认 7.0
        rho: float = 7.0,
        # 求解器的阶数，默认 2
        solver_order: int = 2,
        # 是否进行阈值处理，默认为 False
        thresholding: bool = False,
        # 动态阈值处理的比例，默认 0.995
        dynamic_thresholding_ratio: float = 0.995,
        # 采样的最大值，默认 1.0
        sample_max_value: float = 1.0,
        # 算法类型，默认为 "dpmsolver++"
        algorithm_type: str = "dpmsolver++",
        # 求解器类型，默认为 "midpoint"
        solver_type: str = "midpoint",
        # 最终是否使用较低阶的处理，默认为 True
        lower_order_final: bool = True,
        # 最终步骤是否使用欧拉法，默认为 False
        euler_at_final: bool = False,
        # 最终 sigma 的类型，默认为 "zero"，可选值有 "zero" 和 "sigma_min"
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
    ):
        # DPM-Solver的设置
        # 检查算法类型是否在支持的类型中
        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"]:
            # 如果算法类型是“deis”，则注册为“dpmsolver++”
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            # 否则，抛出未实现的错误
            else:
                raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")

        # 检查求解器类型是否在支持的类型中
        if solver_type not in ["midpoint", "heun"]:
            # 如果求解器类型是“logrho”、“bh1”或“bh2”，则注册为“midpoint”
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            # 否则，抛出未实现的错误
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        # 检查算法类型和最终标准差类型的兼容性
        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"] and final_sigmas_type == "zero":
            # 如果不兼容，抛出值错误
            raise ValueError(
                f"`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please choose `sigma_min` instead."
            )

        # 创建一个从0到1的等间隔张量，长度为训练时间步数
        ramp = torch.linspace(0, 1, num_train_timesteps)
        # 如果sigma调度是“karras”，计算相应的sigma值
        if sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(ramp)
        # 如果sigma调度是“exponential”，计算相应的sigma值
        elif sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(ramp)

        # 对计算得到的sigma进行预处理噪声
        self.timesteps = self.precondition_noise(sigmas)

        # 将sigma与一个零张量连接，确保与设备一致
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        # 可设置的值
        self.num_inference_steps = None  # 推理步骤数量初始化为None
        self.model_outputs = [None] * solver_order  # 模型输出初始化为None列表，长度为求解器顺序
        self.lower_order_nums = 0  # 较低阶数初始化为0
        self._step_index = None  # 步骤索引初始化为None
        self._begin_index = None  # 开始索引初始化为None
        # 将sigma转移到CPU，以避免过多的CPU/GPU通信
        self.sigmas = self.sigmas.to("cpu")

    @property
    def init_noise_sigma(self):
        # 返回初始噪声分布的标准差
        return (self.config.sigma_max**2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度步骤后增加1。
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过管道使用`set_begin_index`方法设置。
        """
        return self._begin_index

    # 从diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index复制
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的开始索引。此函数应在推理之前从管道运行。

        Args:
            begin_index (`int`):
                调度器的开始索引。
        """
        self._begin_index = begin_index

    # 从diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_inputs复制
    def precondition_inputs(self, sample, sigma):
        # 计算输入的预处理系数
        c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
        # 将样本按预处理系数缩放
        scaled_sample = sample * c_in
        # 返回缩放后的样本
        return scaled_sample
    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_noise 复制的
    def precondition_noise(self, sigma):
        # 检查 sigma 是否为 PyTorch 张量，如果不是则转换为张量
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma])

        # 计算 c_noise，基于 sigma 计算对数并乘以 0.25
        c_noise = 0.25 * torch.log(sigma)

        # 返回计算得到的 c_noise
        return c_noise

    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_outputs 复制的
    def precondition_outputs(self, sample, model_output, sigma):
        # 获取配置信息中的 sigma_data
        sigma_data = self.config.sigma_data
        # 计算 c_skip，用于后续的输出合成
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

        # 根据配置中的预测类型计算 c_out
        if self.config.prediction_type == "epsilon":
            # 计算 epsilon 预测下的 c_out
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        elif self.config.prediction_type == "v_prediction":
            # 计算 v 预测下的 c_out
            c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        else:
            # 如果预测类型不支持，则抛出错误
            raise ValueError(f"Prediction type {self.config.prediction_type} is not supported.")

        # 根据 c_skip 和 c_out 组合去噪样本与模型输出
        denoised = c_skip * sample + c_out * model_output

        # 返回去噪后的结果
        return denoised

    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.scale_model_input 复制的
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器的可互换性。通过 `(sigma**2 + 1) ** 0.5` 缩放去噪模型输入，以匹配欧拉算法。

        参数:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。

        返回:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 如果 step_index 尚未初始化，则初始化它
        if self.step_index is None:
            self._init_step_index(timestep)

        # 获取当前 step_index 对应的 sigma
        sigma = self.sigmas[self.step_index]
        # 对输入样本进行预处理
        sample = self.precondition_inputs(sample, sigma)

        # 标记输入缩放函数已被调用
        self.is_scale_input_called = True
        # 返回处理后的样本
        return sample
    # 定义设置推理步骤的方法，接受推理步数和设备类型参数
    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步（在推理之前运行）。

        参数:
            num_inference_steps (`int`):
                用于生成样本的扩散步骤数，使用预训练模型时。
            device (`str` 或 `torch.device`, *可选*):
                时间步应移动到的设备。如果为 `None`，则时间步不会移动。
        """

        # 将推理步骤数保存到实例变量
        self.num_inference_steps = num_inference_steps

        # 生成一个从 0 到 1 的等间距张量，长度为推理步骤数
        ramp = torch.linspace(0, 1, self.num_inference_steps)
        # 根据配置选择 sigma 调度方式
        if self.config.sigma_schedule == "karras":
            # 使用 Karras 方法计算 sigma 值
            sigmas = self._compute_karras_sigmas(ramp)
        elif self.config.sigma_schedule == "exponential":
            # 使用指数方法计算 sigma 值
            sigmas = self._compute_exponential_sigmas(ramp)

        # 将 sigma 转换为浮点类型并移动到指定设备
        sigmas = sigmas.to(dtype=torch.float32, device=device)
        # 对 sigma 进行预处理以获得时间步
        self.timesteps = self.precondition_noise(sigmas)

        # 根据配置决定最后的 sigma 值
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.config.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            # 如果 final_sigmas_type 的值无效，则抛出异常
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        # 将最后的 sigma 值添加到 sigma 张量中
        self.sigmas = torch.cat([sigmas, torch.tensor([sigma_last], dtype=torch.float32, device=device)])

        # 初始化模型输出列表，长度为求解器的阶数
        self.model_outputs = [
            None,
        ] * self.config.solver_order
        # 记录较低阶数的数量
        self.lower_order_nums = 0

        # 为允许重复时间步的调度器添加索引计数器
        self._step_index = None
        self._begin_index = None
        # 将 sigma 移动到 CPU，避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  

    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._compute_karras_sigmas 复制的方法
    def _compute_karras_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        """构建 Karras 等人 (2022) 的噪声调度。"""
        # 如果没有提供 sigma_min，则使用配置中的值
        sigma_min = sigma_min or self.config.sigma_min
        # 如果没有提供 sigma_max，则使用配置中的值
        sigma_max = sigma_max or self.config.sigma_max

        # 获取配置中的 rho 值
        rho = self.config.rho
        # 计算 sigma_min 和 sigma_max 的倒数
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 计算 sigmas，根据 Karras 方法
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._compute_exponential_sigmas 复制的方法
    # 计算指数sigma值，基于给定的ramp和可选的sigma_min和sigma_max参数
    def _compute_exponential_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        # 文档字符串，描述该函数的实现和相关链接
        """Implementation closely follows k-diffusion.
    
        https://github.com/crowsonkb/k-diffusion/blob/6ab5146d4a5ef63901326489f31f1d8e7dd36b48/k_diffusion/sampling.py#L26
        """
        # 如果sigma_min未提供，则使用配置中的默认值
        sigma_min = sigma_min or self.config.sigma_min
        # 如果sigma_max未提供，则使用配置中的默认值
        sigma_max = sigma_max or self.config.sigma_max
        # 生成从log(sigma_min)到log(sigma_max)的等间距值，取指数后翻转顺序
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), len(ramp)).exp().flip(0)
        # 返回计算得到的sigma值
        return sigmas
    
    # 从diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample复制而来
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        # 文档字符串，描述动态阈值处理的实现和效果
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
    
        https://arxiv.org/abs/2205.11487
        """
        # 获取输入样本的数据类型
        dtype = sample.dtype
        # 获取样本的批量大小、通道数和其他维度
        batch_size, channels, *remaining_dims = sample.shape
    
        # 如果数据类型不是float32或float64，则将样本转换为float类型
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half
    
        # 将样本展平，以便沿每个图像进行分位数计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
    
        # 计算样本的绝对值，用于后续的阈值计算
        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"
    
        # 计算样本的动态阈值s，基于配置的动态阈值比例
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 将s限制在[1, sample_max_value]范围内
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        # 在第一维增加一个维度，以便后续广播
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        # 将样本限制在[-s, s]范围内，并除以s进行归一化
        sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"
    
        # 将样本恢复到原始形状
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原始数据类型
        sample = sample.to(dtype)
    
        # 返回处理后的样本
        return sample
    
    # 从diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t复制而来
    # 定义一个私有方法，计算 sigma 和 log_sigmas 之间的关系
    def _sigma_to_t(self, sigma, log_sigmas):
        # 计算 sigma 的对数值，避免过小的 sigma 值引发计算问题
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # 计算 log_sigma 与 log_sigmas 之间的距离
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 找到满足条件的最小索引，表示 sigma 在 log_sigmas 中的范围
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        # 根据索引获取对应的 log_sigmas 值
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 通过线性插值计算权重 w
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)  # 限制 w 的范围在 0 到 1 之间

        # 将插值结果转换为时间范围 t
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)  # 重新调整 t 的形状以匹配 sigma
        return t  # 返回计算出的时间范围

    # 定义一个私有方法，将 sigma 转换为 alpha_t 和 sigma_t
    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = torch.tensor(1)  # 输入在进入 unet 之前已预缩放，因此 alpha_t = 1
        sigma_t = sigma  # 将 sigma 直接赋值给 sigma_t

        return alpha_t, sigma_t  # 返回 alpha_t 和 sigma_t

    # 定义一个方法，将模型输出转换为 DPM 算法需要的类型
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        将模型输出转换为 DPMSolver/DPMSolver++ 算法所需的相应类型。DPM-Solver
        旨在离散化噪声预测模型的积分，DPM-Solver++ 旨在离散化数据预测模型的积分。

        <Tip>

        算法和模型类型是解耦的。您可以使用 DPMSolver 或 DPMSolver++ 处理噪声
        预测和数据预测模型。

        </Tip>

        Args:
            model_output (`torch.Tensor`):
                从学习到的扩散模型直接输出的结果。
            sample (`torch.Tensor`):
                当前由扩散过程创建的样本实例。

        Returns:
            `torch.Tensor`:
                转换后的模型输出。
        """
        sigma = self.sigmas[self.step_index]  # 获取当前步骤的 sigma 值
        x0_pred = self.precondition_outputs(sample, model_output, sigma)  # 预处理模型输出

        if self.config.thresholding:  # 检查是否启用阈值处理
            x0_pred = self._threshold_sample(x0_pred)  # 应用阈值处理

        return x0_pred  # 返回处理后的预测结果

    # 定义一个方法，用于执行 DPM 求解器的第一阶更新
    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        一步操作用于第一阶 DPMSolver（等同于 DDIM）。

        Args:
            model_output (`torch.Tensor`):
                从学习的扩散模型直接输出的张量。
            sample (`torch.Tensor`):
                通过扩散过程创建的当前样本实例。

        Returns:
            `torch.Tensor`:
                上一个时间步的样本张量。
        """
        # 获取当前和上一个时间步的 sigma 值
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        # 将 sigma 转换为 alpha 和 sigma_t
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        # 将上一个 sigma 转换为 alpha 和 sigma_s
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        # 计算 lambda_t 和 lambda_s 的差
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        # 计算 h 值
        h = lambda_t - lambda_s
        # 根据配置选择算法类型并进行相应计算
        if self.config.algorithm_type == "dpmsolver++":
            # 使用 dpmsolver++ 算法计算 x_t
            x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            # 确保噪声不为 None
            assert noise is not None
            # 使用 sde-dpmsolver++ 算法计算 x_t
            x_t = (
                (sigma_t / sigma_s * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )

        # 返回计算得到的样本 x_t
        return x_t

    # 定义多步 DPM Solver 的二阶更新方法
    def multistep_dpm_solver_second_order_update(
        self,
        # 模型输出的张量列表
        model_output_list: List[torch.Tensor],
        # 当前样本的张量，默认为 None
        sample: torch.Tensor = None,
        # 可选的噪声张量，默认为 None
        noise: Optional[torch.Tensor] = None,
    # 返回一个张量，表示第二阶多步 DPMSolver 的一步
    ) -> torch.Tensor:
            """
            一步对于第二阶多步 DPMSolver。
    
            参数：
                model_output_list (`List[torch.Tensor]`):
                    当前及后续时间步学习到的扩散模型直接输出。
                sample (`torch.Tensor`):
                    通过扩散过程生成的当前样本实例。
    
            返回：
                `torch.Tensor`:
                    上一个时间步的样本张量。
            """
            # 获取当前时间步和相邻时间步的 sigma 值
            sigma_t, sigma_s0, sigma_s1 = (
                self.sigmas[self.step_index + 1],  # 下一个时间步的 sigma
                self.sigmas[self.step_index],        # 当前时间步的 sigma
                self.sigmas[self.step_index - 1],   # 上一个时间步的 sigma
            )
    
            # 将 sigma 转换为 alpha 和 sigma_t
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
            alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
            alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
    
            # 计算 lambda 值
            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)  # 当前时间步的 lambda
            lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)  # 上一个时间步的 lambda
            lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)  # 前两个时间步的 lambda
    
            # 获取模型输出的最后两个值
            m0, m1 = model_output_list[-1], model_output_list[-2]
    
            # 计算 h 和 r0
            h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
            r0 = h_0 / h  # 计算 r0
            D0, D1 = m0, (1.0 / r0) * (m0 - m1)  # 计算 D0 和 D1
            if self.config.algorithm_type == "dpmsolver++":  # 检查算法类型
                # 详细推导请参见 https://arxiv.org/abs/2211.01095
                if self.config.solver_type == "midpoint":  # 检查求解器类型
                    # 根据中点法计算 x_t
                    x_t = (
                        (sigma_t / sigma_s0) * sample  # 根据 sigma 调整样本
                        - (alpha_t * (torch.exp(-h) - 1.0)) * D0  # 减去第一个项
                        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1  # 减去第二个项
                    )
                elif self.config.solver_type == "heun":  # 如果使用 Heun 方法
                    # 根据 Heun 方法计算 x_t
                    x_t = (
                        (sigma_t / sigma_s0) * sample  # 根据 sigma 调整样本
                        - (alpha_t * (torch.exp(-h) - 1.0)) * D0  # 减去第一个项
                        + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1  # 加上第二个项
                    )
            elif self.config.algorithm_type == "sde-dpmsolver++":  # 检查另一种算法类型
                assert noise is not None  # 确保噪声不为空
                if self.config.solver_type == "midpoint":  # 使用中点法
                    # 根据中点法计算 x_t，考虑噪声
                    x_t = (
                        (sigma_t / sigma_s0 * torch.exp(-h)) * sample  # 根据 sigma 和样本
                        + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0  # 加上第一个项
                        + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1  # 加上第二个项
                        + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise  # 加上噪声项
                    )
                elif self.config.solver_type == "heun":  # 如果使用 Heun 方法
                    # 根据 Heun 方法计算 x_t，考虑噪声
                    x_t = (
                        (sigma_t / sigma_s0 * torch.exp(-h)) * sample  # 根据 sigma 和样本
                        + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0  # 加上第一个项
                        + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1  # 加上第二个项
                        + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise  # 加上噪声项
                    )
    
            # 返回计算得到的 x_t
            return x_t
    
        # 多步 DPM 求解器的三阶更新
        def multistep_dpm_solver_third_order_update(
            self,
            model_output_list: List[torch.Tensor],  # 模型输出列表
            sample: torch.Tensor = None,  # 当前样本
    ) -> torch.Tensor:  # 指定函数返回值类型为 torch.Tensor
        """  # 文档字符串开始
        One step for the third-order multistep DPMSolver.  # 说明这是第三阶多步 DPMSolver 的一步

        Args:  # 参数说明部分开始
            model_output_list (`List[torch.Tensor]`):  # 输入参数 model_output_list 的类型为列表，元素为 torch.Tensor
                The direct outputs from learned diffusion model at current and latter timesteps.  # 描述该参数为当前及后续时间步的扩散模型输出
            sample (`torch.Tensor`):  # 输入参数 sample 的类型为 torch.Tensor
                A current instance of a sample created by diffusion process.  # 描述该参数为扩散过程中创建的当前样本实例

        Returns:  # 返回值说明部分开始
            `torch.Tensor`:  # 返回值类型为 torch.Tensor
                The sample tensor at the previous timestep.  # 描述返回的是上一个时间步的样本张量
        """  # 文档字符串结束
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (  # 从 sigmas 属性中提取当前及前两步的 sigma 值
            self.sigmas[self.step_index + 1],  # 获取下一个时间步的 sigma 值
            self.sigmas[self.step_index],  # 获取当前时间步的 sigma 值
            self.sigmas[self.step_index - 1],  # 获取前一个时间步的 sigma 值
            self.sigmas[self.step_index - 2],  # 获取前两个时间步的 sigma 值
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)  # 将 sigma_t 转换为对应的 alpha_t 和 sigma_t
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)  # 将 sigma_s0 转换为对应的 alpha_s0 和 sigma_s0
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)  # 将 sigma_s1 转换为对应的 alpha_s1 和 sigma_s1
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)  # 将 sigma_s2 转换为对应的 alpha_s2 和 sigma_s2

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)  # 计算 lambda_t，表示 alpha_t 和 sigma_t 的对数差
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)  # 计算 lambda_s0，表示 alpha_s0 和 sigma_s0 的对数差
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)  # 计算 lambda_s1，表示 alpha_s1 和 sigma_s1 的对数差
        lambda_s2 = torch.log(alpha_s2) - torch.log(sigma_s2)  # 计算 lambda_s2，表示 alpha_s2 和 sigma_s2 的对数差

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]  # 从输出列表中提取最近三次模型输出

        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2  # 计算 h 值，表示各时间步的差异
        r0, r1 = h_0 / h, h_1 / h  # 计算 r0 和 r1，表示比例关系
        D0 = m0  # 赋值 D0 为最近的模型输出 m0
        D1_0, D1_1 = (1.0 / r0) * (m0 - m1), (1.0 / r1) * (m1 - m2)  # 计算 D1_0 和 D1_1，表示调整后的输出差异
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)  # 计算 D1，结合 D1_0 和 D1_1 的信息
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)  # 计算 D2，表示更高阶的差异

        if self.config.algorithm_type == "dpmsolver++":  # 检查算法类型是否为 "dpmsolver++"
            # See https://arxiv.org/abs/2206.00927 for detailed derivations  # 参考文献，提供详细推导信息
            x_t = (  # 计算当前时间步的样本张量 x_t
                (sigma_t / sigma_s0) * sample  # 计算样本的加权项
                - (alpha_t * (torch.exp(-h) - 1.0)) * D0  # 减去 D0 的调整项
                + (alpha_t * ((torch.exp(-h) - 1.0) / h + 1.0)) * D1  # 添加 D1 的调整项
                - (alpha_t * ((torch.exp(-h) - 1.0 + h) / h**2 - 0.5)) * D2  # 减去 D2 的调整项
            )

        return x_t  # 返回计算得到的样本张量 x_t

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep  # 注明此行代码来源于特定的调度器
    # 根据时间步（timestep）获取对应的索引
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果没有指定调度时间步，则使用实例的时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 查找与给定时间步匹配的所有候选索引
        index_candidates = (schedule_timesteps == timestep).nonzero()

        # 如果没有找到候选索引，则设置为最后一个时间步的索引
        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # 如果找到多个候选索引，则取第二个索引（确保不跳过）
        # 这样可以确保在去噪计划中间开始时不会跳过任何 sigma
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        # 否则，取第一个候选索引
        else:
            step_index = index_candidates[0].item()

        # 返回找到的时间步索引
        return step_index

    # 从调度器的 DPMSolverMultistepScheduler 初始化步骤索引的函数
    def _init_step_index(self, timestep):
        """
        初始化调度器的步骤索引计数器。
        """

        # 如果开始索引为空
        if self.begin_index is None:
            # 如果时间步是一个张量，则将其转换到与时间步相同的设备上
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            # 通过调用 index_for_timestep 函数初始化步骤索引
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 否则，将步骤索引设置为开始索引
            self._step_index = self._begin_index

    # 执行单步操作的函数
    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    # 从 EulerDiscreteScheduler 的 add_noise 函数复制的代码
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    # 返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 的设备和数据类型与 original_samples 相同
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 如果设备为 mps 且 timesteps 是浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            # 将 timesteps 转换为相同设备和 float32 类型
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 在其他情况下，将 schedule_timesteps 转换为 original_samples 的设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            # 将 timesteps 转换为 original_samples 的设备
            timesteps = timesteps.to(original_samples.device)
    
        # 当 scheduler 用于训练或 pipeline 未实现 set_begin_index 时，self.begin_index 为 None
        if self.begin_index is None:
            # 获取每个 timesteps 对应的步骤索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise 在第一次去噪步骤之后被调用（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一次去噪步骤之前调用 add noise，以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]
    
        # 根据步骤索引获取 sigma，并将其展平
        sigma = sigmas[step_indices].flatten()
        # 扩展 sigma 的形状，以匹配 original_samples 的维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
    
        # 将噪声添加到原始样本中，生成噪声样本
        noisy_samples = original_samples + noise * sigma
        # 返回噪声样本
        return noisy_samples
    
    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```