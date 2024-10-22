# `.\diffusers\schedulers\scheduling_dpmsolver_singlestep.py`

```py
# 版权所有 2024 TSAIL Team 和 HuggingFace Team。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是在“按现状”基础上分发的，
# 不提供任何形式的保证或条件。
# 请参阅许可证以了解管理权限和
# 限制的具体条款。

# 声明：此文件受 https://github.com/LuChengTHU/dpm-solver 的强烈影响

import math  # 导入数学库以进行数学计算
from typing import List, Optional, Tuple, Union  # 导入类型提示以支持类型检查

import numpy as np  # 导入 NumPy 库以支持数组操作
import torch  # 导入 PyTorch 库以支持张量运算

from ..configuration_utils import ConfigMixin, register_to_config  # 导入配置混合器和注册函数
from ..utils import deprecate, logging  # 导入弃用和日志记录工具
from ..utils.torch_utils import randn_tensor  # 导入生成随机张量的工具
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput  # 导入调度相关工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，便于调试


# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 输入参数：扩散时间步数
    max_beta=0.999,  # 输入参数：最大 beta 值，默认值为 0.999
    alpha_transform_type="cosine",  # 输入参数：alpha 转换类型，默认值为 "cosine"
):
    """
    创建一个 beta 调度，离散化给定的 alpha_t_bar 函数，该函数定义了
    随时间的 (1-beta) 的累积乘积，从 t = [0,1]。

    包含一个函数 alpha_bar，它接受一个参数 t，并将其转换为
    扩散过程的 (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以
                     防止奇异性。
        alpha_transform_type (`str`, *可选*，默认为 `cosine`): alpha_bar 的噪声调度类型。
                     可选择 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于逐步模型输出的 betas
    """
    if alpha_transform_type == "cosine":  # 如果 alpha 转换类型为 "cosine"

        def alpha_bar_fn(t):  # 定义 alpha_bar 函数
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # 计算 alpha_bar 值

    elif alpha_transform_type == "exp":  # 如果 alpha 转换类型为 "exp"

        def alpha_bar_fn(t):  # 定义 alpha_bar 函数
            return math.exp(t * -12.0)  # 计算 alpha_bar 值

    else:  # 如果 alpha 转换类型不支持
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")  # 抛出异常

    betas = []  # 初始化一个空列表以存储 beta 值
    for i in range(num_diffusion_timesteps):  # 遍历每个扩散时间步
        t1 = i / num_diffusion_timesteps  # 计算当前时间步 t1
        t2 = (i + 1) / num_diffusion_timesteps  # 计算下一个时间步 t2
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))  # 计算 beta 并添加到列表
    return torch.tensor(betas, dtype=torch.float32)  # 返回作为张量的 beta 列表


class DPMSolverSinglestepScheduler(SchedulerMixin, ConfigMixin):  # 定义一个类，继承调度器混合器和配置混合器
    """
    `DPMSolverSinglestepScheduler` 是一个快速的专用高阶求解器，用于扩散 ODE。
    # 该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]，查看父类文档以获取库为所有调度器实现的通用方法，如加载和保存。
    
        _compatibles = [e.name for e in KarrasDiffusionSchedulers]  # 获取 KarrasDiffusionSchedulers 中所有调度器的名称
        order = 1  # 设置调度器的初始顺序为 1
    
        @register_to_config  # 将该方法注册到配置中
        def __init__(  # 初始化方法
            self,
            num_train_timesteps: int = 1000,  # 训练的时间步数，默认值为 1000
            beta_start: float = 0.0001,  # 起始 beta 值，默认值为 0.0001
            beta_end: float = 0.02,  # 结束 beta 值，默认值为 0.02
            beta_schedule: str = "linear",  # beta 值的调度方式，默认值为线性
            trained_betas: Optional[np.ndarray] = None,  # 可选的已训练 beta 值
            solver_order: int = 2,  # 解算器的顺序，默认值为 2
            prediction_type: str = "epsilon",  # 预测类型，默认值为 'epsilon'
            thresholding: bool = False,  # 是否应用阈值处理，默认值为 False
            dynamic_thresholding_ratio: float = 0.995,  # 动态阈值比例，默认值为 0.995
            sample_max_value: float = 1.0,  # 生成样本的最大值，默认值为 1.0
            algorithm_type: str = "dpmsolver++",  # 算法类型，默认值为 'dpmsolver++'
            solver_type: str = "midpoint",  # 解算器类型，默认值为 'midpoint'
            lower_order_final: bool = False,  # 最终是否使用较低的顺序，默认值为 False
            use_karras_sigmas: Optional[bool] = False,  # 是否使用 Karras 的 sigma，默认值为 False
            final_sigmas_type: Optional[str] = "zero",  # 最终 sigma 的类型，默认为 'zero'
            lambda_min_clipped: float = -float("inf"),  # 最小 lambda 值，默认为负无穷
            variance_type: Optional[str] = None,  # 方差类型，默认为 None
        def get_order_list(self, num_inference_steps: int) -> List[int]:  # 获取每个时间步的解算器顺序
            """
            计算每个时间步的解算器顺序。
    
            Args:
                num_inference_steps (`int`):  # 输入参数：生成样本时使用的扩散步数
                    生成样本时使用的扩散步数。
            """
            steps = num_inference_steps  # 将输入的扩散步数赋值给变量 steps
            order = self.config.solver_order  # 从配置中获取解算器顺序
            if order > 3:  # 如果解算器顺序大于 3
                raise ValueError("Order > 3 is not supported by this scheduler")  # 抛出异常，表示不支持
            if self.config.lower_order_final:  # 如果配置要求最终使用较低的顺序
                if order == 3:  # 如果解算器顺序为 3
                    if steps % 3 == 0:  # 如果步数能够被 3 整除
                        orders = [1, 2, 3] * (steps // 3 - 1) + [1, 2] + [1]  # 按照顺序生成 orders 列表
                    elif steps % 3 == 1:  # 如果步数除以 3 的余数为 1
                        orders = [1, 2, 3] * (steps // 3) + [1]  # 生成 orders 列表
                    else:  # 如果步数除以 3 的余数为 2
                        orders = [1, 2, 3] * (steps // 3) + [1, 2]  # 生成 orders 列表
                elif order == 2:  # 如果解算器顺序为 2
                    if steps % 2 == 0:  # 如果步数能够被 2 整除
                        orders = [1, 2] * (steps // 2 - 1) + [1, 1]  # 生成 orders 列表
                    else:  # 如果步数不能被 2 整除
                        orders = [1, 2] * (steps // 2) + [1]  # 生成 orders 列表
                elif order == 1:  # 如果解算器顺序为 1
                    orders = [1] * steps  # 生成 orders 列表
            else:  # 如果最终不使用较低的顺序
                if order == 3:  # 如果解算器顺序为 3
                    orders = [1, 2, 3] * (steps // 3)  # 生成 orders 列表
                elif order == 2:  # 如果解算器顺序为 2
                    orders = [1, 2] * (steps // 2)  # 生成 orders 列表
                elif order == 1:  # 如果解算器顺序为 1
                    orders = [1] * steps  # 生成 orders 列表
            return orders  # 返回生成的 orders 列表
    
        @property
        def step_index(self):  # 当前时间步的索引计数器属性
            """
            当前时间步的索引计数器。每次调度器步骤后增加 1。
            """
            return self._step_index  # 返回当前步骤索引
    
        @property
        def begin_index(self):  # 第一个时间步的索引属性
            """
            第一个时间步的索引。应该通过 `set_begin_index` 方法从管道中设置。
            """
            return self._begin_index  # 返回开始步骤索引
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler 复制的 set_begin_index 方法
    # 设置调度器的起始索引，默认值为 0
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的起始索引。此函数应在推理前从管道中运行。

        Args:
            begin_index (`int`):
                调度器的起始索引。
        """
        # 将传入的起始索引赋值给实例变量
        self._begin_index = begin_index

    # 设置推理步骤的数量、设备和时间步
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "动态阈值：在每个采样步骤中，我们将 s 设置为 xt0（t 时刻 x_0 的预测）的某个百分位绝对像素值，
        如果 s > 1，则将 xt0 阈值设定在范围 [-s, s] 内，然后除以 s。动态阈值在每一步主动防止像素饱和， 
        我们发现动态阈值显著提高了照片现实主义以及图像-文本对齐，特别是在使用非常大的指导权重时。"

        https://arxiv.org/abs/2205.11487
        """
        # 获取样本的数值类型
        dtype = sample.dtype
        # 解包样本的形状，获取批大小、通道和剩余维度
        batch_size, channels, *remaining_dims = sample.shape

        # 如果数据类型不是 float32 或 float64，则将其转换为 float
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 为分位数计算上溯，且未实现对 CPU half 的限制

        # 将样本展平以便在每个图像上进行分位数计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        # 计算样本的绝对值
        abs_sample = sample.abs()  # "某个百分位绝对像素值"

        # 计算样本绝对值的给定分位数
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 限制 s 的范围，最小值为 1，最大值为配置的样本最大值
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当限制为最小值 1 时，相当于标准限制在 [-1, 1] 的范围内
        # 扩展 s 的维度以便广播
        s = s.unsqueeze(1)  # (batch_size, 1) 因为 clamp 会在 dim=0 上广播
        # 将样本限制在范围 [-s, s] 内并除以 s
        sample = torch.clamp(sample, -s, s) / s  # "我们将 xt0 阈值设定在范围 [-s, s] 内，然后除以 s"

        # 将样本的形状还原回原来的维度
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原始数据类型
        sample = sample.to(dtype)

        # 返回处理后的样本
        return sample

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制
    # 定义一个私有方法，将 sigma 和 log_sigmas 转换为时间 t
    def _sigma_to_t(self, sigma, log_sigmas):
        # 计算 log sigma，防止 sigma 为零，使用 np.maximum 限制最小值
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # 计算分布，得到 log_sigma 和 log_sigmas 之间的距离
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 获取 sigmas 的范围，找到低索引和高索引
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        # 根据低索引和高索引提取对应的 log_sigmas 值
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 进行 sigma 的插值计算
        w = (low - log_sigma) / (low - high)
        # 限制 w 的范围在 [0, 1]
        w = np.clip(w, 0, 1)

        # 将插值转换为时间范围 t
        t = (1 - w) * low_idx + w * high_idx
        # 调整 t 的形状以匹配 sigma 的形状
        t = t.reshape(sigma.shape)
        # 返回计算得到的时间 t
        return t

    # 定义一个私有方法，将 sigma 转换为 alpha_t 和 sigma_t
    def _sigma_to_alpha_sigma_t(self, sigma):
        # 计算 alpha_t，表示在时间 t 的方差
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        # 计算 sigma_t，结合 alpha_t 调整 sigma
        sigma_t = sigma * alpha_t

        # 返回计算得到的 alpha_t 和 sigma_t
        return alpha_t, sigma_t

    # 定义一个私有方法，将输入 sigmas 转换为 Karras 的噪声调度
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构建 Karras 等人 (2022) 的噪声调度。"""

        # 确保其他调度器在复制此函数时不会出错的 Hack
        # TODO: 将此逻辑添加到其他调度器中
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min  # 获取 sigma_min 配置
        else:
            sigma_min = None  # 如果没有配置，则为 None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max  # 获取 sigma_max 配置
        else:
            sigma_max = None  # 如果没有配置，则为 None

        # 确保 sigma_min 和 sigma_max 的值有效
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 论文中使用的常数值
        # 创建一个从 0 到 1 的 ramp，用于插值
        ramp = np.linspace(0, 1, num_inference_steps)
        # 计算 sigma_min 和 sigma_max 的倒数
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 计算最终的 sigmas 值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回计算得到的 sigmas
        return sigmas

    # 定义一个方法，将模型输出进行转换
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    # 定义一个方法，进行 DPM 求解器的一阶更新
    def dpm_solver_first_order_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    # 定义函数，返回一个张量，表示一阶 DPMSolver 的一步操作（相当于 DDIM）
    ) -> torch.Tensor:
            """
            一阶 DPMSolver 的一步，等价于 DDIM。
    
            参数：
                model_output (`torch.Tensor`):
                    从学习到的扩散模型直接输出的张量。
                timestep (`int`):
                    当前扩散链中的离散时间步。
                prev_timestep (`int`):
                    上一个离散时间步。
                sample (`torch.Tensor`):
                    由扩散过程创建的当前样本实例。
    
            返回：
                `torch.Tensor`:
                    在上一个时间步的样本张量。
            """
            # 从参数中提取当前时间步，若未提供则为 None
            timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
            # 从参数中提取上一个时间步，若未提供则为 None
            prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
            # 若样本为 None，则尝试从参数中获取样本
            if sample is None:
                if len(args) > 2:
                    sample = args[2]
                else:
                    # 若仍未获取样本，则抛出错误
                    raise ValueError(" missing `sample` as a required keyward argument")
            # 若提供当前时间步，进行弃用警告
            if timestep is not None:
                deprecate(
                    "timesteps",
                    "1.0.0",
                    "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
                )
    
            # 若提供上一个时间步，进行弃用警告
            if prev_timestep is not None:
                deprecate(
                    "prev_timestep",
                    "1.0.0",
                    "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
                )
            # 获取当前和上一个时间步对应的 sigma 值
            sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
            # 将 sigma 转换为 alpha 和 sigma_t
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
            # 将 sigma 转换为 alpha 和 sigma_s
            alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
            # 计算 lambda_t 和 lambda_s
            lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
            lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
            # 计算 h 值
            h = lambda_t - lambda_s
            # 根据算法类型计算 x_t
            if self.config.algorithm_type == "dpmsolver++":
                x_t = (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * model_output
            elif self.config.algorithm_type == "dpmsolver":
                x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output
            elif self.config.algorithm_type == "sde-dpmsolver++":
                # 确保噪声不为 None
                assert noise is not None
                x_t = (
                    (sigma_t / sigma_s * torch.exp(-h)) * sample
                    + (alpha_t * (1 - torch.exp(-2.0 * h))) * model_output
                    + sigma_t * torch.sqrt(1.0 - torch.exp(-2 * h)) * noise
                )
            # 返回计算得到的 x_t
            return x_t
    
        # 定义第二阶更新的单步 DPM 求解器函数
        def singlestep_dpm_solver_second_order_update(
            self,
            # 接受模型输出列表，类型为张量列表
            model_output_list: List[torch.Tensor],
            # 接受可变参数
            *args,
            # 可选的样本张量，默认为 None
            sample: torch.Tensor = None,
            # 可选的噪声张量，默认为 None
            noise: Optional[torch.Tensor] = None,
            # 接受其他关键字参数
            **kwargs,
    # 定义第三阶更新的单步 DPM 求解器
        def singlestep_dpm_solver_third_order_update(
            self,
            model_output_list: List[torch.Tensor],  # 模型输出列表，包含当前和后续时间步的输出
            *args,  # 可变位置参数
            sample: torch.Tensor = None,  # 当前样本，默认为 None
            **kwargs,  # 可变关键字参数
        def singlestep_dpm_solver_update(
            self,
            model_output_list: List[torch.Tensor],  # 模型输出列表
            *args,  # 可变位置参数
            sample: torch.Tensor = None,  # 当前样本，默认为 None
            order: int = None,  # 当前步骤的求解器阶数
            noise: Optional[torch.Tensor] = None,  # 噪声张量，默认为 None
            **kwargs,  # 可变关键字参数
        ) -> torch.Tensor:  # 返回类型为 torch.Tensor
            """
            单步执行单步 DPM 求解器的更新。
    
            参数:
                model_output_list (`List[torch.Tensor]`):
                    当前和后续时间步的学习扩散模型的直接输出。
                timestep (`int`):
                    当前和后续的离散时间步。
                prev_timestep (`int`):
                    前一个离散时间步。
                sample (`torch.Tensor`):
                    扩散过程创建的当前样本实例。
                order (`int`):
                    当前步骤的求解器阶数。
    
            返回:
                `torch.Tensor`:
                    前一个时间步的样本张量。
            """
            # 获取时间步列表，如果没有则从关键字参数中取出
            timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
            # 获取前一个时间步，如果没有则从关键字参数中取出
            prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
            # 如果样本为 None，尝试从参数中获取或抛出异常
            if sample is None:
                if len(args) > 2:
                    sample = args[2]  # 从参数中获取样本
                else:
                    raise ValueError(" missing`sample` as a required keyward argument")  # 抛出异常
            # 如果阶数为 None，尝试从参数中获取或抛出异常
            if order is None:
                if len(args) > 3:
                    order = args[3]  # 从参数中获取阶数
                else:
                    raise ValueError(" missing `order` as a required keyward argument")  # 抛出异常
            # 如果时间步列表不为 None，发出弃用警告
            if timestep_list is not None:
                deprecate(
                    "timestep_list",  # 弃用的参数名称
                    "1.0.0",  # 弃用版本
                    "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",  # 弃用原因
                )
    
            # 如果前一个时间步不为 None，发出弃用警告
            if prev_timestep is not None:
                deprecate(
                    "prev_timestep",  # 弃用的参数名称
                    "1.0.0",  # 弃用版本
                    "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",  # 弃用原因
                )
    
            # 根据阶数调用相应的更新方法并返回结果
            if order == 1:
                return self.dpm_solver_first_order_update(model_output_list[-1], sample=sample, noise=noise)  # 一阶更新
            elif order == 2:
                return self.singlestep_dpm_solver_second_order_update(model_output_list, sample=sample, noise=noise)  # 二阶更新
            elif order == 3:
                return self.singlestep_dpm_solver_third_order_update(model_output_list, sample=sample)  # 三阶更新
            else:
                raise ValueError(f"Order must be 1, 2, 3, got {order}")  # 抛出异常，阶数无效
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep 复制的代码
    # 定义一个根据时间步查找索引的函数
        def index_for_timestep(self, timestep, schedule_timesteps=None):
            # 如果没有提供调度时间步，则使用类的时间步
            if schedule_timesteps is None:
                schedule_timesteps = self.timesteps
    
            # 查找调度时间步中与当前时间步匹配的索引候选
            index_candidates = (schedule_timesteps == timestep).nonzero()
    
            # 如果没有匹配的索引候选，则取最后一个时间步的索引
            if len(index_candidates) == 0:
                step_index = len(self.timesteps) - 1
            # 如果找到多个匹配，则选择第二个匹配的索引
            # 这样可以确保在去噪调度的中间开始时不会跳过 sigma
            elif len(index_candidates) > 1:
                step_index = index_candidates[1].item()
            # 否则，选择第一个匹配的索引
            else:
                step_index = index_candidates[0].item()
    
            # 返回找到的步索引
            return step_index
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index 复制的函数
        def _init_step_index(self, timestep):
            """
            初始化调度器的 step_index 计数器。
            """
    
            # 如果开始索引未设置
            if self.begin_index is None:
                # 如果时间步是张量，将其移动到时间步所在的设备
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.to(self.timesteps.device)
                # 使用 index_for_timestep 方法获取当前的步索引
                self._step_index = self.index_for_timestep(timestep)
            else:
                # 否则，使用已设置的开始索引
                self._step_index = self._begin_index
    
        # 定义 step 方法，处理模型输出和其他参数
        def step(
            self,
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
            generator=None,
            return_dict: bool = True,
    # 定义一个函数，用于预测上一个时间步的样本，反向传播 SDE
    ) -> Union[SchedulerOutput, Tuple]:
            """
            预测上一个时间步的样本，通过反向 SDE 实现。该函数使用单步 DPMSolver 传播样本。
    
            参数:
                model_output (`torch.Tensor`):
                    来自学习的扩散模型的直接输出。
                timestep (`int`):
                    当前扩散链中的离散时间步。
                sample (`torch.Tensor`):
                    扩散过程中创建的当前样本实例。
                return_dict (`bool`):
                    是否返回 [`~schedulers.scheduling_utils.SchedulerOutput`] 或 `tuple`。
    
            返回:
                [`~schedulers.scheduling_utils.SchedulerOutput`] 或 `tuple`:
                    如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_utils.SchedulerOutput`]，否则返回一个
                    元组，元组的第一个元素是样本张量。
    
            """
            # 检查推断步骤数量是否为 None
            if self.num_inference_steps is None:
                # 如果为 None，则抛出错误，提示需要设置时间步
                raise ValueError(
                    "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
                )
    
            # 如果步骤索引为 None，初始化步骤索引
            if self.step_index is None:
                self._init_step_index(timestep)
    
            # 转换模型输出以适应当前样本
            model_output = self.convert_model_output(model_output, sample=sample)
            # 将模型输出向前移动以存储最新的模型输出
            for i in range(self.config.solver_order - 1):
                self.model_outputs[i] = self.model_outputs[i + 1]
            # 将最新的模型输出存储在最后一个位置
            self.model_outputs[-1] = model_output
    
            # 如果算法类型是 "sde-dpmsolver++"，则生成噪声
            if self.config.algorithm_type == "sde-dpmsolver++":
                noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            else:
                # 否则将噪声设为 None
                noise = None
    
            # 获取当前步骤的顺序
            order = self.order_list[self.step_index]
    
            # 对于 img2img，去噪时可能从 order>1 开始，需确保前两步均为 order=1
            while self.model_outputs[-order] is None:
                order -= 1
    
            # 对于单步求解器，使用每个时间步的初始值，顺序为 1
            if order == 1:
                self.sample = sample
    
            # 使用单步 DPM 求解器更新样本
            prev_sample = self.singlestep_dpm_solver_update(
                self.model_outputs, sample=self.sample, order=order, noise=noise
            )
    
            # 完成后将步骤索引加一，噪声为噪声
            self._step_index += 1
    
            # 如果不返回字典，则返回一个只包含 prev_sample 的元组
            if not return_dict:
                return (prev_sample,)
    
            # 否则返回 SchedulerOutput 对象
            return SchedulerOutput(prev_sample=prev_sample)
    
        # 定义一个函数，用于缩放模型输入
        def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            """
            确保与需要根据当前时间步缩放去噪模型输入的调度器互换性。
    
            参数:
                sample (`torch.Tensor`):
                    输入样本。
    
            返回:
                `torch.Tensor`:
                    一个缩放后的输入样本。
            """
            # 直接返回输入样本
            return sample
    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.add_noise 复制而来
    def add_noise(
            self,
            original_samples: torch.Tensor,  # 输入的原始样本，类型为张量
            noise: torch.Tensor,  # 要添加的噪声，类型为张量
            timesteps: torch.IntTensor,  # 时间步，类型为整数张量
    ) -> torch.Tensor:  # 函数返回添加噪声后的样本，类型为张量
            # 确保 sigmas 和 timesteps 与 original_samples 具有相同的设备和数据类型
            sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
            if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):  # 检查是否在 mps 设备上且 timesteps 是浮点类型
                # mps 不支持 float64 类型
                schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)  # 将 timesteps 转换为 float32
                timesteps = timesteps.to(original_samples.device, dtype=torch.float32)  # 将 timesteps 转换为 float32
            else:
                schedule_timesteps = self.timesteps.to(original_samples.device)  # 将 timesteps 转换为与 original_samples 相同的设备
                timesteps = timesteps.to(original_samples.device)  # 将 timesteps 转换为与 original_samples 相同的设备
    
            # 当调度器用于训练或管道未实现 set_begin_index 时，begin_index 为 None
            if self.begin_index is None:
                # 根据给定的 timesteps 计算相应的步骤索引
                step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
            elif self.step_index is not None:
                # 在第一次去噪步骤之后调用 add_noise（用于修复）
                step_indices = [self.step_index] * timesteps.shape[0]  # 使用当前步骤索引
            else:
                # 在第一次去噪步骤之前调用 add_noise 以创建初始潜在图像（img2img）
                step_indices = [self.begin_index] * timesteps.shape[0]  # 使用开始索引
    
            sigma = sigmas[step_indices].flatten()  # 获取相应步骤的 sigma 并展平
            while len(sigma.shape) < len(original_samples.shape):  # 确保 sigma 的维度与 original_samples 匹配
                sigma = sigma.unsqueeze(-1)  # 在最后一个维度增加一维
    
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)  # 将 sigma 转换为 alpha_t 和 sigma_t
            noisy_samples = alpha_t * original_samples + sigma_t * noise  # 计算添加噪声后的样本
            return noisy_samples  # 返回添加噪声后的样本
    
        def __len__(self):  # 定义获取对象长度的方法
            return self.config.num_train_timesteps  # 返回训练时间步的数量
```