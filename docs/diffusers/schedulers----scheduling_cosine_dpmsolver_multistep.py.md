# `.\diffusers\schedulers\scheduling_cosine_dpmsolver_multistep.py`

```py
# 版权声明，说明此文件属于 TSAIL 团队和 HuggingFace 团队，所有权利保留
# 
# 根据 Apache 许可证 2.0 版（“许可证”）进行许可；
# 除非遵循许可证，否则您不能使用此文件。
# 您可以在以下网址获取许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是按“现状”提供的，
# 不附带任何形式的担保或条件，无论是明示还是暗示。
# 请参阅许可证以获取有关特定语言治理权限和
# 限制的详细信息。

# 免责声明：此文件受 https://github.com/LuChengTHU/dpm-solver 和 https://github.com/NVlabs/edm 的强烈影响

# 导入数学库以便进行数学计算
import math
# 从 typing 模块导入类型提示相关的类型
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库，提供数组操作和数学函数
import numpy as np
# 导入 PyTorch 库，用于深度学习和张量计算
import torch

# 从配置工具导入配置混合类和注册到配置的装饰器
from ..configuration_utils import ConfigMixin, register_to_config
# 从调度模块导入布朗树噪声采样器
from .scheduling_dpmsolver_sde import BrownianTreeNoiseSampler
# 从调度工具导入调度混合类和调度输出类
from .scheduling_utils import SchedulerMixin, SchedulerOutput


# 定义一个名为 CosineDPMSolverMultistepScheduler 的调度器类
class CosineDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    实现了一种带余弦调度的 `DPMSolverMultistepScheduler` 变体，由 Nichol 和 Dhariwal (2021) 提出。
    此调度器用于 Stable Audio Open [1]。

    [1] Evans, Parker 等人. "Stable Audio Open" https://arxiv.org/abs/2407.14358

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。
    请查看超类文档以获取库为所有调度器实现的通用方法，例如加载和保存。
    # 参数说明
    Args:
        # 噪声调度中的最小噪声幅度，默认值为 0.3
        sigma_min (`float`, *optional*, defaults to 0.3):
            Minimum noise magnitude in the sigma schedule. This was set to 0.3 in Stable Audio Open [1].
        # 噪声调度中的最大噪声幅度，默认值为 500
        sigma_max (`float`, *optional*, defaults to 500):
            Maximum noise magnitude in the sigma schedule. This was set to 500 in Stable Audio Open [1].
        # 数据分布的标准差，默认值为 1.0
        sigma_data (`float`, *optional*, defaults to 1.0):
            The standard deviation of the data distribution. This is set to 1.0 in Stable Audio Open [1].
        # 用于计算 `sigmas` 的调度方式，默认使用指数调度
        sigma_schedule (`str`, *optional*, defaults to `exponential`):
            Sigma schedule to compute the `sigmas`. By default, we the schedule introduced in the EDM paper
            (https://arxiv.org/abs/2206.00364). Other acceptable value is "exponential". The exponential schedule was
            incorporated in this model: https://huggingface.co/stabilityai/cosxl.
        # 训练模型的扩散步骤数量，默认值为 1000
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        # DPMSolver 的顺序，默认值为 2
        solver_order (`int`, defaults to 2):
            The DPMSolver order which can be `1` or `2`. It is recommended to use `solver_order=2`.
        # 调度函数的预测类型，默认值为 `v_prediction`
        prediction_type (`str`, defaults to `v_prediction`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        # 第二阶求解器的类型，默认值为 `midpoint`
        solver_type (`str`, defaults to `midpoint`):
            Solver type for the second-order solver; can be `midpoint` or `heun`. The solver type slightly affects the
            sample quality, especially for a small number of steps. It is recommended to use `midpoint` solvers.
        # 最后步骤中是否使用低阶求解器，默认值为 True
        lower_order_final (`bool`, defaults to `True`):
            Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
            stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
        # 最后一步中是否使用欧拉法，默认值为 False
        euler_at_final (`bool`, defaults to `False`):
            Whether to use Euler's method in the final step. It is a trade-off between numerical stability and detail
            richness. This can stabilize the sampling of the SDE variant of DPMSolver for small number of inference
            steps, but sometimes may result in blurring.
        # 采样过程中噪声调度的最终 `sigma` 值，默认值为 `"zero"`
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
    """

    # 初始化兼容列表
    _compatibles = []
    # 初始化求解器顺序，默认值为 1
    order = 1

    # 注册到配置
    @register_to_config
    # 初始化类的构造函数，设置默认参数
        def __init__(
            # 最小标准差，默认为0.3
            sigma_min: float = 0.3,
            # 最大标准差，默认为500
            sigma_max: float = 500,
            # 数据标准差，默认为1.0
            sigma_data: float = 1.0,
            # 标准差调度方式，默认为"exponential"
            sigma_schedule: str = "exponential",
            # 训练时间步数，默认为1000
            num_train_timesteps: int = 1000,
            # 求解器的阶数，默认为2
            solver_order: int = 2,
            # 预测类型，默认为"v_prediction"
            prediction_type: str = "v_prediction",
            # ρ值，默认为7.0
            rho: float = 7.0,
            # 求解器类型，默认为"midpoint"
            solver_type: str = "midpoint",
            # 是否在最后使用低阶求解器，默认为True
            lower_order_final: bool = True,
            # 是否在最后使用欧拉法，默认为False
            euler_at_final: bool = False,
            # 最终标准差类型，默认为"zero"
            final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
        ):
            # 检查求解器类型是否有效
            if solver_type not in ["midpoint", "heun"]:
                # 如果求解器类型是"logrho"、"bh1"或"bh2"，注册为"midpoint"
                if solver_type in ["logrho", "bh1", "bh2"]:
                    self.register_to_config(solver_type="midpoint")
                # 如果求解器类型无效，抛出未实现错误
                else:
                    raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")
    
            # 创建从0到1的线性间隔，长度为训练时间步数
            ramp = torch.linspace(0, 1, num_train_timesteps)
            # 根据调度类型计算标准差
            if sigma_schedule == "karras":
                sigmas = self._compute_karras_sigmas(ramp)
            elif sigma_schedule == "exponential":
                sigmas = self._compute_exponential_sigmas(ramp)
    
            # 预处理噪声
            self.timesteps = self.precondition_noise(sigmas)
    
            # 将计算得到的标准差与零张量连接
            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
    
            # 可设值初始化
            self.num_inference_steps = None  # 推理步骤数初始化为None
            self.model_outputs = [None] * solver_order  # 根据求解器阶数初始化输出列表
            self.lower_order_nums = 0  # 低阶数初始化为0
            self._step_index = None  # 步骤索引初始化为None
            self._begin_index = None  # 开始索引初始化为None
            # 将标准差张量移到CPU以减少CPU/GPU间通信
            self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication
    
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
            第一个时间步的索引。应通过`set_begin_index`方法从管道设置。
            """
            return self._begin_index
    
        # 从diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler复制的设置开始索引方法
        def set_begin_index(self, begin_index: int = 0):
            """
            设置调度器的开始索引。此函数应在推理前从管道运行。
    
            参数:
                begin_index (`int`):
                    调度器的开始索引。
            """
            self._begin_index = begin_index  # 设置开始索引
    
        # 从diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler复制的预处理输入方法
        def precondition_inputs(self, sample, sigma):
            # 计算输入样本的缩放因子
            c_in = 1 / ((sigma**2 + self.config.sigma_data**2) ** 0.5)
            # 对样本进行缩放处理
            scaled_sample = sample * c_in
            return scaled_sample  # 返回缩放后的样本
    
        def precondition_noise(self, sigma):
            # 如果sigma不是张量，则将其转换为张量
            if not isinstance(sigma, torch.Tensor):
                sigma = torch.tensor([sigma])
    
            # 返回噪声预处理结果
            return sigma.atan() / math.pi * 2  # 计算预处理噪声并返回
    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_outputs 复制而来
    def precondition_outputs(self, sample, model_output, sigma):
        # 获取配置中的 sigma 数据
        sigma_data = self.config.sigma_data
        # 计算 c_skip 值，用于后续的去噪过程
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    
        # 根据预测类型选择不同的计算方式
        if self.config.prediction_type == "epsilon":
            # 计算 c_out 值，使用 epsilon 预测类型
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        elif self.config.prediction_type == "v_prediction":
            # 计算 c_out 值，使用 v_prediction 预测类型
            c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        else:
            # 如果预测类型不支持，抛出异常
            raise ValueError(f"Prediction type {self.config.prediction_type} is not supported.")
    
        # 计算去噪后的结果
        denoised = c_skip * sample + c_out * model_output
    
        # 返回去噪后的样本
        return denoised
    
    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.scale_model_input 复制而来
    def scale_model_input(self, sample: torch.Tensor, timestep: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器的互换性。
        按 `(sigma**2 + 1) ** 0.5` 缩放去噪模型输入，以匹配 Euler 算法。
    
        参数:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *optional*):
                扩散链中的当前时间步。
    
        返回:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 如果步骤索引为 None，初始化步骤索引
        if self.step_index is None:
            self._init_step_index(timestep)
    
        # 获取当前步骤对应的 sigma 值
        sigma = self.sigmas[self.step_index]
        # 预处理输入样本
        sample = self.precondition_inputs(sample, sigma)
    
        # 标记输入缩放已被调用
        self.is_scale_input_called = True
        # 返回预处理后的样本
        return sample
    # 定义设置离散时间步的方法，接受推理步骤数和设备作为参数
    def set_timesteps(self, num_inference_steps: int = None, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步（在推理之前运行）。
    
        参数：
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数，与预训练模型结合。
            device (`str` 或 `torch.device`, *可选*):
                时间步应该移动到的设备。如果为 `None`，则时间步不移动。
        """
    
        # 将输入的推理步骤数赋值给类属性
        self.num_inference_steps = num_inference_steps
    
        # 创建一个从 0 到 1 的线性空间，步数为 num_inference_steps
        ramp = torch.linspace(0, 1, self.num_inference_steps)
        
        # 根据配置选择不同的 sigma 计算方式
        if self.config.sigma_schedule == "karras":
            # 计算 Karras 方法的 sigma 值
            sigmas = self._compute_karras_sigmas(ramp)
        elif self.config.sigma_schedule == "exponential":
            # 计算指数方法的 sigma 值
            sigmas = self._compute_exponential_sigmas(ramp)
    
        # 将 sigma 转换为 float32 类型，并移动到指定设备
        sigmas = sigmas.to(dtype=torch.float32, device=device)
        
        # 对 sigma 进行预处理以获取时间步
        self.timesteps = self.precondition_noise(sigmas)
    
        # 根据配置选择最后一个 sigma 值的类型
        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.config.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            # 抛出异常，如果 final_sigmas_type 不是预期的类型
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )
    
        # 将计算得到的 sigma 和最后的 sigma 拼接
        self.sigmas = torch.cat([sigmas, torch.tensor([sigma_last], dtype=torch.float32, device=device)])
    
        # 初始化模型输出列表，长度为配置的求解器顺序
        self.model_outputs = [
            None,
        ] * self.config.solver_order
        
        # 初始化较低阶数目
        self.lower_order_nums = 0
    
        # 为允许重复时间步的调度器添加索引计数器
        self._step_index = None
        self._begin_index = None
        
        # 将 sigma 移动到 CPU，避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")
    
        # 如果使用噪声采样器，重新初始化它
        self.noise_sampler = None
    
    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._compute_karras_sigmas 复制而来
    def _compute_karras_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        """构建 Karras 等人的噪声调度（2022）。"""
        
        # 如果 sigma_min 和 sigma_max 为 None，则使用配置中的默认值
        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max
    
        # 获取配置中的 rho 值
        rho = self.config.rho
        
        # 计算 sigma_min 和 sigma_max 的倒数
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        
        # 根据 ramp 计算 sigmas 的值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas
    
    # 从 diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._compute_exponential_sigmas 复制而来
    # 计算指数 sigma 值，基于 k-diffusion 的实现
    def _compute_exponential_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> torch.Tensor:
        # 如果 sigma_min 为 None，使用配置中的 sigma_min
        sigma_min = sigma_min or self.config.sigma_min
        # 如果 sigma_max 为 None，使用配置中的 sigma_max
        sigma_max = sigma_max or self.config.sigma_max
        # 创建从 sigma_min 到 sigma_max 的对数等间隔值，计算其指数并反转顺序
        sigmas = torch.linspace(math.log(sigma_min), math.log(sigma_max), len(ramp)).exp().flip(0)
        # 返回计算出的 sigma 值
        return sigmas
    
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制的函数
    def _sigma_to_t(self, sigma, log_sigmas):
        # 获取 sigma 的对数值，确保不小于 1e-10
        log_sigma = np.log(np.maximum(sigma, 1e-10))
        # 计算 log_sigma 与 log_sigmas 之间的距离
        dists = log_sigma - log_sigmas[:, np.newaxis]
        # 计算 sigmas 范围的低索引和高索引
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1
        # 获取低和高的 log_sigma 值
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]
        # 计算加权值用于插值
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)
        # 将插值转化为时间范围
        t = (1 - w) * low_idx + w * high_idx
        # 调整 t 的形状以匹配 sigma 的形状
        t = t.reshape(sigma.shape)
        # 返回计算出的时间 t
        return t
    
    def _sigma_to_alpha_sigma_t(self, sigma):
        # 设置 alpha_t 为 1，因为输入在进入 unet 之前已经预先缩放
        alpha_t = torch.tensor(1)  # Inputs are pre-scaled before going into unet, so alpha_t = 1
        # sigma_t 直接赋值为输入的 sigma
        sigma_t = sigma
        # 返回 alpha_t 和 sigma_t
        return alpha_t, sigma_t
    
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        将模型输出转换为 DPMSolver/DPMSolver++ 算法所需的相应类型。DPM-Solver 旨在离散化噪声预测模型的积分，
        而 DPM-Solver++ 则旨在离散化数据预测模型的积分。
    
        <Tip>
        算法和模型类型是解耦的。您可以使用 DPMSolver 或 DPMSolver++ 处理噪声预测和数据预测模型。
        </Tip>
    
        Args:
            model_output (`torch.Tensor`):
                从学习到的扩散模型直接输出的张量。
            sample (`torch.Tensor`):
                扩散过程中创建的当前样本实例。
    
        Returns:
            `torch.Tensor`:
                转换后的模型输出。
        """
        # 根据当前步骤索引获取对应的 sigma 值
        sigma = self.sigmas[self.step_index]
        # 预处理模型输出，得到预测的 x0 值
        x0_pred = self.precondition_outputs(sample, model_output, sigma)
        # 返回预测结果 x0_pred
        return x0_pred
    
    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        # 获取当前步骤和下一个步骤的噪声水平
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        # 将噪声水平转换为 alpha 和 sigma
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        # 计算当前和前一步的 lambda 值
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

        # 计算 h 值，表示两个 lambda 之间的差异
        h = lambda_t - lambda_s
        # 确保噪声不为 None，进行断言检查
        assert noise is not None
        # 根据当前样本、模型输出和噪声计算新的样本 x_t
        x_t = (
            (sigma_t / sigma_s * torch.exp(-h)) * sample
            + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
            + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
        )

        # 返回计算得到的样本 x_t
        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[torch.Tensor],
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        一步执行二阶多步DPMSolver。

        参数：
            model_output_list (`List[torch.Tensor]`):
                当前和后续时间步长从学习到的扩散模型的直接输出。
            sample (`torch.Tensor`):
                由扩散过程创建的当前样本实例。

        返回：
            `torch.Tensor`:
                前一个时间步的样本张量。
        """
        # 从 sigma 列表中获取当前、前一和前两步的 sigma 值
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        # 将 sigma 转换为 alpha 和 sigma_t
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        # 计算 lambda 值，用于后续计算
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

        # 从模型输出列表中获取最后两个输出
        m0, m1 = model_output_list[-1], model_output_list[-2]

        # 计算 h 和 h_0，用于噪声调整
        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h  # 计算比例 r0
        D0, D1 = m0, (1.0 / r0) * (m0 - m1)  # 计算 D0 和 D1

        # sde-dpmsolver++
        assert noise is not None  # 确保噪声不为空
        if self.config.solver_type == "midpoint":
            # 使用 midpoint 方法计算 x_t
            x_t = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                + 0.5 * (alpha_t * (1 - torch.exp(-2.0 * h))) * D1
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )
        elif self.config.solver_type == "heun":
            # 使用 heun 方法计算 x_t
            x_t = (
                (sigma_t / sigma_s0 * torch.exp(-h)) * sample
                + (alpha_t * (1 - torch.exp(-2.0 * h))) * D0
                + (alpha_t * ((1.0 - torch.exp(-2.0 * h)) / (-2.0 * h) + 1.0)) * D1
                + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
            )

        return x_t  # 返回计算得到的 x_t

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep 复制的代码
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果没有提供调度时间步，使用默认时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        # 找到与当前时间步匹配的索引候选
        index_candidates = (schedule_timesteps == timestep).nonzero()

        # 如果没有找到匹配，设置为最后一个时间步
        if len(index_candidates) == 0:
            step_index = len(self.timesteps) - 1
        # 如果找到多个匹配，选择第二个索引
        elif len(index_candidates) > 1:
            step_index = index_candidates[1].item()
        else:
            # 否则选择第一个匹配的索引
            step_index = index_candidates[0].item()

        return step_index  # 返回计算得到的步索引
    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index 复制而来
    def _init_step_index(self, timestep):
        """
        初始化调度器的 step_index 计数器。
        """

        # 如果 begin_index 为 None，表示尚未设置开始索引
        if self.begin_index is None:
            # 检查 timestep 是否为 PyTorch 的张量
            if isinstance(timestep, torch.Tensor):
                # 将 timestep 转换到与 timesteps 设备相同
                timestep = timestep.to(self.timesteps.device)
            # 根据当前的 timestep 计算并设置 step_index
            self._step_index = self.index_for_timestep(timestep)
        else:
            # 如果 begin_index 已经设置，则直接使用它
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise 复制而来
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 的设备和数据类型与 original_samples 相同
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 如果设备为 mps 且 timesteps 是浮点类型
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps 不支持 float64 类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            # 将 timesteps 转换为与 original_samples 相同的设备和数据类型
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将 schedule_timesteps 转换为与 original_samples 相同的设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            # 将 timesteps 转换为与 original_samples 相同的设备
            timesteps = timesteps.to(original_samples.device)

        # 当 scheduler 用于训练时，或者管道未实现 set_begin_index 时，begin_index 为 None
        if self.begin_index is None:
            # 计算每个 timesteps 对应的 step indices
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise 在第一次去噪步骤后被调用（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise 在第一次去噪步骤之前被调用以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 获取与当前 step_indices 相关联的 sigma 值，并将其展平
        sigma = sigmas[step_indices].flatten()
        # 扩展 sigma 以匹配 original_samples 的形状
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 添加噪声到原始样本
        noisy_samples = original_samples + noise * sigma
        # 返回带噪声的样本
        return noisy_samples

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```