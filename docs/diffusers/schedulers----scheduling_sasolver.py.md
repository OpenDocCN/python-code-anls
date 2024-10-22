# `.\diffusers\schedulers\scheduling_sasolver.py`

```py
# 版权所有 2024 Shuchen Xue 等人在中国科学院大学团队和 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件
# 根据许可证分发是以“原样”基础进行的，
# 不附带任何形式的明示或暗示的担保或条件。
# 请参见许可证以获取特定语言管理权限和
# 限制条款。

# 免责声明：请查看 https://arxiv.org/abs/2309.05019
# 该代码库是基于 https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py 的修改版本

# 导入数学库，用于数学运算
import math
# 从 typing 模块导入所需的类型注解
from typing import Callable, List, Optional, Tuple, Union

# 导入 numpy 库，用于数组和数学运算
import numpy as np
# 导入 torch 库，用于张量计算
import torch

# 从配置工具模块导入配置混合类和注册配置函数
from ..configuration_utils import ConfigMixin, register_to_config
# 从 utils 模块导入弃用警告工具
from ..utils import deprecate
# 从 torch_utils 模块导入生成随机张量的函数
from ..utils.torch_utils import randn_tensor
# 从调度工具模块导入调度器类和输出类
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


# 从 diffusers.schedulers.scheduling_ddpm 导入 betas_for_alpha_bar 函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 需要生成的 beta 数量
    max_beta=0.999,  # 最大 beta 值，使用小于 1 的值以防止奇点
    alpha_transform_type="cosine",  # alpha 变换类型，默认为余弦
):
    """
    创建一个 beta 调度程序，该程序离散化给定的 alpha_t_bar 函数，该函数定义了
    随时间推移 (1-beta) 的累积乘积，从 t = [0,1] 开始。

    包含一个 alpha_bar 函数，该函数接受参数 t 并将其转换为 (1-beta) 的累积乘积
    直到扩散过程的那部分。


    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 要使用的最大 beta；使用小于 1 的值以
                     防止奇点。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     从 `cosine` 或 `exp` 中选择

    返回：
        betas (`np.ndarray`): 调度程序用于更新模型输出的 betas
    """
    # 如果 alpha 变换类型为余弦
    if alpha_transform_type == "cosine":
        # 定义 alpha_bar 函数，返回余弦平方值
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 如果 alpha 变换类型为指数
    elif alpha_transform_type == "exp":
        # 定义 alpha_bar 函数，返回指数衰减值
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    # 如果 alpha 变换类型不支持，则引发错误
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []  # 初始化 betas 列表
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps  # 当前时间步归一化
        t2 = (i + 1) / num_diffusion_timesteps  # 下一个时间步归一化
        # 计算 beta 值并添加到列表中，限制最大值
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 返回以 torch 张量形式表示的 betas
    return torch.tensor(betas, dtype=torch.float32)


# 定义一个新的类 SASolverScheduler，继承自 SchedulerMixin 和 ConfigMixin
class SASolverScheduler(SchedulerMixin, ConfigMixin):
    """
    `SASolverScheduler` 是一个快速的专用高阶求解器，用于扩散 SDEs。
    # 该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]，查看超类文档以了解库为所有调度器实现的通用方法，例如加载和保存。
    
        _compatibles = [e.name for e in KarrasDiffusionSchedulers]  # 创建一个包含 KarrasDiffusionSchedulers 中所有调度器名称的列表
        order = 1  # 设置调度器的默认顺序为 1
    
        @register_to_config  # 将该初始化方法注册到配置中
        def __init__(  # 定义初始化方法
            self,  # 当前实例
            num_train_timesteps: int = 1000,  # 训练时的时间步数，默认为 1000
            beta_start: float = 0.0001,  # beta 参数的起始值，默认为 0.0001
            beta_end: float = 0.02,  # beta 参数的结束值，默认为 0.02
            beta_schedule: str = "linear",  # beta 调度类型，默认为线性
            trained_betas: Optional[Union[np.ndarray, List[float]]] = None,  # 可选的训练 beta 值，默认为 None
            predictor_order: int = 2,  # 预测器的阶数，默认为 2
            corrector_order: int = 2,  # 校正器的阶数，默认为 2
            prediction_type: str = "epsilon",  # 预测类型，默认为 "epsilon"
            tau_func: Optional[Callable] = None,  # 可选的 tau 函数，默认为 None
            thresholding: bool = False,  # 是否使用阈值处理，默认为 False
            dynamic_thresholding_ratio: float = 0.995,  # 动态阈值比例，默认为 0.995
            sample_max_value: float = 1.0,  # 采样的最大值，默认为 1.0
            algorithm_type: str = "data_prediction",  # 算法类型，默认为 "data_prediction"
            lower_order_final: bool = True,  # 是否使用较低阶数的最终步骤，默认为 True
            use_karras_sigmas: Optional[bool] = False,  # 是否使用 Karras 的 sigma 值，默认为 False
            lambda_min_clipped: float = -float("inf"),  # lambda 的最小裁剪值，默认为负无穷大
            variance_type: Optional[str] = None,  # 可选的方差类型，默认为 None
            timestep_spacing: str = "linspace",  # 时间步间隔类型，默认为线性间隔
            steps_offset: int = 0,  # 时间步偏移量，默认为 0
    ):
        # 如果训练好的 beta 参数不为 None，则将其转换为浮点张量
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果 beta_schedule 为 "linear"，则生成线性间隔的 beta 值
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果 beta_schedule 为 "scaled_linear"，则按特定方式生成 beta 值
        elif beta_schedule == "scaled_linear":
            # 此调度非常特定于潜在扩散模型
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2  # 平方以得到最终的 beta 值
            )
        # 如果 beta_schedule 为 "squaredcos_cap_v2"，则使用 Glide 余弦调度生成 beta 值
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果 beta_schedule 不匹配任何已知类型，则抛出未实现的错误
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alpha 值为 1 减去对应的 beta 值
        self.alphas = 1.0 - self.betas
        # 计算 alpha 值的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 当前仅支持 VP 类型的噪声调度
        self.alpha_t = torch.sqrt(self.alphas_cumprod)  # 计算当前时刻的 alpha 值的平方根
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)  # 计算当前时刻的噪声标准差
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)  # 计算 lambda 值
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5  # 计算 sigma 值

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 检查算法类型是否有效，若无效则抛出未实现的错误
        if algorithm_type not in ["data_prediction", "noise_prediction"]:
            raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")

        # 可设定的值
        self.num_inference_steps = None  # 推理步骤数初始化为 None
        # 创建时间步数组，从 0 到 num_train_timesteps-1 的线性间隔
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        self.timesteps = torch.from_numpy(timesteps)  # 将时间步数组转换为张量
        # 初始化时间步列表和模型输出列表
        self.timestep_list = [None] * max(predictor_order, corrector_order - 1)
        self.model_outputs = [None] * max(predictor_order, corrector_order - 1)

        # 如果 tau_func 为 None，设置默认的 tau 函数
        if tau_func is None:
            self.tau_func = lambda t: 1 if t >= 200 and t <= 800 else 0
        else:
            self.tau_func = tau_func  # 否则使用传入的 tau 函数
        # 根据算法类型确定是否预测 x0
        self.predict_x0 = algorithm_type == "data_prediction"
        self.lower_order_nums = 0  # 较低阶数初始化为 0
        self.last_sample = None  # 上一个样本初始化为 None
        self._step_index = None  # 当前步骤索引初始化为 None
        self._begin_index = None  # 起始步骤索引初始化为 None
        self.sigmas = self.sigmas.to("cpu")  # 将 sigmas 移动到 CPU，以减少 CPU/GPU 之间的通信

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加 1。
        """
        return self._step_index  # 返回当前步骤索引

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过 `set_begin_index` 方法从管道设置。
        """
        return self._begin_index  # 返回起始步骤索引

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制
    # 设置调度器的起始索引，默认值为0
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的起始索引。此函数应在推理前从管道运行。
    
        Args:
            begin_index (`int`):
                调度器的起始索引。
        """
        # 将提供的起始索引赋值给内部变量
        self._begin_index = begin_index
    
        # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制的代码
        def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
            """
            "动态阈值处理：在每个采样步骤中，我们将 s 设置为 xt0 中的某个百分位绝对像素值（在时间步 t 的 x_0 预测），
            如果 s > 1，则将 xt0 限制在范围 [-s, s]，然后除以 s。动态阈值处理将饱和像素（接近 -1 和 1 的像素）推向内部，
            从而在每一步中积极防止像素饱和。我们发现动态阈值处理显著改善了照片现实感以及图像文本对齐，尤其是在使用非常大的引导权重时。"
    
            https://arxiv.org/abs/2205.11487
            """
            # 获取样本的数值类型
            dtype = sample.dtype
            # 获取批大小、通道数和其他维度
            batch_size, channels, *remaining_dims = sample.shape
    
            # 如果样本数据类型不是 float32 或 float64，则将其转换为 float
            if dtype not in (torch.float32, torch.float64):
                sample = sample.float()  # 为百分位数计算向上转换，并且对 CPU 半精度未实现限制
    
            # 将样本展平以进行每幅图像的百分位数计算
            sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
    
            # 获取样本的绝对值
            abs_sample = sample.abs()  # "某个百分位绝对像素值"
    
            # 计算绝对样本的指定百分位数
            s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
            # 限制 s 的范围，以避免其小于1或超过最大值
            s = torch.clamp(
                s, min=1, max=self.config.sample_max_value
            )  # 当限制到 min=1 时，相当于标准剪切到 [-1, 1]
            # 扩展 s 以适应批处理
            s = s.unsqueeze(1)  # (batch_size, 1) 因为限制会在维度0上广播
            # 限制样本并将其缩放到范围 [-s, s]
            sample = torch.clamp(sample, -s, s) / s  # "将 xt0 限制在范围 [-s, s] 并除以 s"
    
            # 将样本形状恢复为原始形状
            sample = sample.reshape(batch_size, channels, *remaining_dims)
            # 将样本转换回原始数据类型
            sample = sample.to(dtype)
    
            # 返回处理后的样本
            return sample
    
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制的代码
        def _sigma_to_t(self, sigma, log_sigmas):
            # 获取 sigma 的对数值
            log_sigma = np.log(np.maximum(sigma, 1e-10))
    
            # 获取分布
            dists = log_sigma - log_sigmas[:, np.newaxis]
    
            # 获取 sigma 的范围
            low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
    
            # 获取低和高对数 sigma 值
            low = log_sigmas[low_idx]
            high = log_sigmas[high_idx]
    
            # 插值计算 sigma
            w = (low - log_sigma) / (low - high)
            w = np.clip(w, 0, 1)
    
            # 将插值转换为时间范围
            t = (1 - w) * low_idx + w * high_idx
            # 将 t 的形状调整为 sigma 的形状
            t = t.reshape(sigma.shape)
            # 返回计算得到的 t 值
            return t
    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._sigma_to_alpha_sigma_t 拷贝的函数
    def _sigma_to_alpha_sigma_t(self, sigma):
        # 根据 sigma 计算 alpha_t，公式为 1 / √(sigma² + 1)
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        # 计算 sigma_t，公式为 sigma * alpha_t
        sigma_t = sigma * alpha_t

        # 返回 alpha_t 和 sigma_t
        return alpha_t, sigma_t

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras 拷贝的函数
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构建 Karras 等人 (2022) 的噪声调度。"""

        # 确保其他拷贝此函数的调度器不会出现错误的黑客方法
        # TODO: 将此逻辑添加到其他调度器中
        if hasattr(self.config, "sigma_min"):
            # 如果配置中有 sigma_min，则使用该值
            sigma_min = self.config.sigma_min
        else:
            # 否则将 sigma_min 设为 None
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            # 如果配置中有 sigma_max，则使用该值
            sigma_max = self.config.sigma_max
        else:
            # 否则将 sigma_max 设为 None
            sigma_max = None

        # 如果 sigma_min 为 None，则取 in_sigmas 的最后一个值
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        # 如果 sigma_max 为 None，则取 in_sigmas 的第一个值
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        # 设置 rho 值为 7.0，参考文献中的用法
        rho = 7.0  
        # 创建一个从 0 到 1 的线性 ramp
        ramp = np.linspace(0, 1, num_inference_steps)
        # 计算 min_inv_rho 和 max_inv_rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 计算 sigmas，使用 ramp 插值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回计算得到的 sigmas
        return sigmas

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    # 计算指数负函数在给定区间的积分
    def get_coefficients_exponential_negative(self, order, interval_start, interval_end):
        """
        计算从 interval_start 到 interval_end 的积分 exp(-x) * x^order dx
        """
        # 确保 order 只支持 0, 1, 2 和 3
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        # 如果 order 为 0，使用对应的公式计算并返回结果
        if order == 0:
            return torch.exp(-interval_end) * (torch.exp(interval_end - interval_start) - 1)
        # 如果 order 为 1，使用对应的公式计算并返回结果
        elif order == 1:
            return torch.exp(-interval_end) * (
                (interval_start + 1) * torch.exp(interval_end - interval_start) - (interval_end + 1)
            )
        # 如果 order 为 2，使用对应的公式计算并返回结果
        elif order == 2:
            return torch.exp(-interval_end) * (
                (interval_start**2 + 2 * interval_start + 2) * torch.exp(interval_end - interval_start)
                - (interval_end**2 + 2 * interval_end + 2)
            )
        # 如果 order 为 3，使用对应的公式计算并返回结果
        elif order == 3:
            return torch.exp(-interval_end) * (
                (interval_start**3 + 3 * interval_start**2 + 6 * interval_start + 6)
                * torch.exp(interval_end - interval_start)
                - (interval_end**3 + 3 * interval_end**2 + 6 * interval_end + 6)
            )
    # 计算给定区间内与指数相关的积分系数
    def get_coefficients_exponential_positive(self, order, interval_start, interval_end, tau):
        """
        计算从 interval_start 到 interval_end 的积分
        公式为 exp(x(1+tau^2)) * x^order dx
        """
        # 确保 order 的值只在支持的范围内
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        # 变量替换后计算结束区间
        interval_end_cov = (1 + tau**2) * interval_end
        # 变量替换后计算起始区间
        interval_start_cov = (1 + tau**2) * interval_start

        # 处理 order 为 0 的情况
        if order == 0:
            return (
                # 计算并返回积分结果
                torch.exp(interval_end_cov) * (1 - torch.exp(-(interval_end_cov - interval_start_cov))) / (1 + tau**2)
            )
        # 处理 order 为 1 的情况
        elif order == 1:
            return (
                # 计算并返回积分结果
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov - 1)
                    - (interval_start_cov - 1) * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 2)
            )
        # 处理 order 为 2 的情况
        elif order == 2:
            return (
                # 计算并返回积分结果
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov**2 - 2 * interval_end_cov + 2)
                    - (interval_start_cov**2 - 2 * interval_start_cov + 2)
                    * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 3)
            )
        # 处理 order 为 3 的情况
        elif order == 3:
            return (
                # 计算并返回积分结果
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov**3 - 3 * interval_end_cov**2 + 6 * interval_end_cov - 6)
                    - (interval_start_cov**3 - 3 * interval_start_cov**2 + 6 * interval_start_cov - 6)
                    * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 4)
            )

    # 计算系数的函数
    def get_coefficients_fn(self, order, interval_start, interval_end, lambda_list, tau):
        # 确保 order 的值在支持的范围内
        assert order in [1, 2, 3, 4]
        # 确保 lambda_list 的长度等于 order
        assert order == len(lambda_list), "the length of lambda list must be equal to the order"
        # 初始化系数列表
        coefficients = []
        # 计算拉格朗日多项式系数
        lagrange_coefficient = self.lagrange_polynomial_coefficient(order - 1, lambda_list)
        # 遍历 order 的范围
        for i in range(order):
            # 初始化单个系数
            coefficient = 0
            # 遍历 order 的范围
            for j in range(order):
                # 根据预测标志选择不同的积分计算方法
                if self.predict_x0:
                    # 计算正指数的系数
                    coefficient += lagrange_coefficient[i][j] * self.get_coefficients_exponential_positive(
                        order - 1 - j, interval_start, interval_end, tau
                    )
                else:
                    # 计算负指数的系数
                    coefficient += lagrange_coefficient[i][j] * self.get_coefficients_exponential_negative(
                        order - 1 - j, interval_start, interval_end
                    )
            # 将计算得到的系数添加到列表中
            coefficients.append(coefficient)
        # 确保系数列表的长度与 order 匹配
        assert len(coefficients) == order, "the length of coefficients does not match the order"
        # 返回计算得到的系数列表
        return coefficients
    # 定义随机亚当斯-巴什福斯更新函数
        def stochastic_adams_bashforth_update(
            self,
            model_output: torch.Tensor,  # 模型输出的张量
            *args,  # 额外的参数，使用可变参数
            sample: torch.Tensor,  # 输入样本的张量
            noise: torch.Tensor,  # 噪声的张量
            order: int,  # 更新的顺序
            tau: torch.Tensor,  # 时间步长的张量
            **kwargs,  # 额外的关键字参数
        # 定义随机亚当斯-莫尔顿更新函数
        def stochastic_adams_moulton_update(
            self,
            this_model_output: torch.Tensor,  # 当前模型输出的张量
            *args,  # 额外的参数，使用可变参数
            last_sample: torch.Tensor,  # 上一个样本的张量
            last_noise: torch.Tensor,  # 上一个噪声的张量
            this_sample: torch.Tensor,  # 当前样本的张量
            order: int,  # 更新的顺序
            tau: torch.Tensor,  # 时间步长的张量
            **kwargs,  # 额外的关键字参数
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep 复制的函数
        def index_for_timestep(self, timestep, schedule_timesteps=None):
            # 如果未提供调度时间步，使用默认时间步
            if schedule_timesteps is None:
                schedule_timesteps = self.timesteps
    
            # 查找与当前时间步相匹配的索引候选
            index_candidates = (schedule_timesteps == timestep).nonzero()
    
            # 如果没有找到匹配的候选索引
            if len(index_candidates) == 0:
                # 设置步索引为时间步列表的最后一个索引
                step_index = len(self.timesteps) - 1
            # 如果找到多个匹配的候选索引
            # 第一个步骤的 sigma 索引总是取第二个索引（或只有一个时取最后一个索引）
            # 这样可以确保在去噪声调度中间开始时不会意外跳过一个 sigma
            elif len(index_candidates) > 1:
                # 取第二个候选索引作为步索引
                step_index = index_candidates[1].item()
            else:
                # 取第一个候选索引作为步索引
                step_index = index_candidates[0].item()
    
            # 返回步索引
            return step_index
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index 复制的函数
        def _init_step_index(self, timestep):
            """
            初始化调度器的 step_index 计数器。
            """
    
            # 如果 begin_index 为空
            if self.begin_index is None:
                # 如果时间步是张量类型，将其转换为与时间步设备一致
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.to(self.timesteps.device)
                # 根据时间步获取步索引
                self._step_index = self.index_for_timestep(timestep)
            else:
                # 使用预定义的开始索引
                self._step_index = self._begin_index
    
        # 定义步函数
        def step(
            self,
            model_output: torch.Tensor,  # 模型输出的张量
            timestep: int,  # 当前时间步
            sample: torch.Tensor,  # 输入样本的张量
            generator=None,  # 可选的生成器
            return_dict: bool = True,  # 是否返回字典
        # 定义缩放模型输入的函数
        def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            """
            确保与需要根据当前时间步缩放去噪声模型输入的调度器的可互换性。
    
            参数：
                sample (`torch.Tensor`):
                    输入样本。
    
            返回：
                `torch.Tensor`:
                    缩放后的输入样本。
            """
            # 直接返回输入样本
            return sample
    
        # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的函数
        def add_noise(
            self,
            original_samples: torch.Tensor,  # 原始样本的张量
            noise: torch.Tensor,  # 噪声的张量
            timesteps: torch.IntTensor,  # 时间步的整数张量
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 与 original_samples 具有相同的设备和数据类型
        # 将 self.alphas_cumprod 移动到目标设备，以避免后续 add_noise 调用时冗余的 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为 original_samples 的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到 original_samples 的设备
        timesteps = timesteps.to(original_samples.device)

        # 计算 sqrt_alpha_prod 为 alphas_cumprod 在 timesteps 索引下的平方根
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平为一维张量
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度少于 original_samples 的维度，则在最后一维添加一个新的维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 sqrt_one_minus_alpha_prod 为 1 减去 alphas_cumprod 在 timesteps 索引下的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平为一维张量
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度少于 original_samples 的维度，则在最后一维添加一个新的维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 计算加噪声的样本，将原始样本与噪声按加权组合
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回加噪声后的样本
        return noisy_samples

    # 定义获取对象长度的方法
    def __len__(self):
        # 返回配置中定义的训练时间步数
        return self.config.num_train_timesteps
```