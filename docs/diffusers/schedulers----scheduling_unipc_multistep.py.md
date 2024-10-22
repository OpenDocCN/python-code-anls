# `.\diffusers\schedulers\scheduling_unipc_multistep.py`

```py
# 版权信息，指明作者及团队
# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
#
# 许可信息，指明使用此文件的条件
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 说明如何获取许可证
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 说明法律责任和条件
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证的具体权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 免责声明，指向相关信息链接
# DISCLAIMER: check https://arxiv.org/abs/2302.04867 and https://github.com/wl-zhao/UniPC for more info
# 此代码基于指定 GitHub 项目进行修改
# The codebase is modified based on https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py

# 导入数学模块
import math
# 从 typing 模块导入所需的类型
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch

# 从配置工具导入相关混合类和配置注册函数
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具模块导入弃用函数
from ..utils import deprecate
# 从调度工具导入所需的调度类和输出类
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


# 定义函数，生成 beta 调度
# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建 beta 调度，以离散化给定的 alpha_t_bar 函数，该函数定义了
    (1-beta) 的累积乘积，时间范围从 t = [0,1]。

    包含一个 alpha_bar 函数，该函数接受参数 t，并将其转换为
    在扩散过程中到该部分的 (1-beta) 的累积乘积。

    参数:
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用低于 1 的值以
                     防止奇点。
        alpha_transform_type (`str`, *optional*, default to `cosine`): alpha_bar 的噪声调度类型。
                     可选择 `cosine` 或 `exp`

    返回:
        betas (`np.ndarray`): 调度器用于步骤模型输出的 betas
    """
    # 根据 alpha_transform_type 确定 alpha_bar 函数
    if alpha_transform_type == "cosine":
        # 定义余弦变换的 alpha_bar 函数
        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":
        # 定义指数变换的 alpha_bar 函数
        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        # 抛出不支持的类型错误
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化 beta 列表
    betas = []
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算并添加 beta 值，限制在 max_beta 之下
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 返回以 tensor 格式的 beta 值
    return torch.tensor(betas, dtype=torch.float32)


# 定义函数，重新缩放终止信噪比
# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    # 重新缩放 betas，使其具有零终端信噪比，参考文献 https://arxiv.org/pdf/2305.08891.pdf (算法 1)

    # 参数:
    #     betas (`torch.Tensor`):
    #         用于初始化调度器的 betas。

    # 返回:
    #     `torch.Tensor`: 具有零终端信噪比的重新缩放后的 betas
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas，表示每个时间步的 alpha 值
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算累积乘积的平方根

    # 存储旧值。
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 记录第一个时间步的平方根值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 记录最后一个时间步的平方根值

    # 进行平移，使最后一个时间步为零。
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 从所有值中减去最后一个值

    # 进行缩放，使第一个时间步恢复为旧值。
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 缩放以恢复第一个值

    # 将 alphas_bar_sqrt 转换回 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方根，得到 alphas_bar
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 还原累积乘积，计算每个时间步的 alpha
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个 alpha 添加到结果中
    betas = 1 - alphas  # 计算 betas，作为 1 减去 alphas

    return betas  # 返回重新缩放后的 betas
# 定义一个名为 UniPCMultistepScheduler 的类，继承自 SchedulerMixin 和 ConfigMixin
class UniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `UniPCMultistepScheduler` 是一个无训练的框架，旨在快速采样扩散模型。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关库为所有调度程序实现的通用
    方法（例如加载和保存）的文档，请查看超类文档。
    """

    # 定义与 KarrasDiffusionSchedulers 兼容的名称列表
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置调度程序的顺序为 1
    order = 1

    # 使用 @register_to_config 装饰器将该方法注册到配置
    @register_to_config
    def __init__(
        # 设置训练时间步的数量，默认值为 1000
        num_train_timesteps: int = 1000,
        # 设置 beta 的起始值，默认值为 0.0001
        beta_start: float = 0.0001,
        # 设置 beta 的结束值，默认值为 0.02
        beta_end: float = 0.02,
        # 设置 beta 的调度方式，默认为线性
        beta_schedule: str = "linear",
        # 训练好的 beta 值，可选，默认为 None
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 设置求解器的阶数，默认值为 2
        solver_order: int = 2,
        # 设置预测类型，默认为 epsilon
        prediction_type: str = "epsilon",
        # 是否启用阈值处理，默认为 False
        thresholding: bool = False,
        # 设置动态阈值处理比例，默认值为 0.995
        dynamic_thresholding_ratio: float = 0.995,
        # 设置样本的最大值，默认值为 1.0
        sample_max_value: float = 1.0,
        # 是否预测 x0，默认为 True
        predict_x0: bool = True,
        # 设置求解器类型，默认值为 "bh2"
        solver_type: str = "bh2",
        # 是否在最后一步降低阶数，默认为 True
        lower_order_final: bool = True,
        # 禁用校正器的步骤列表，默认为空
        disable_corrector: List[int] = [],
        # 设置求解器 p，默认为 None
        solver_p: SchedulerMixin = None,
        # 是否使用 Karras 的 sigma 值，可选，默认为 False
        use_karras_sigmas: Optional[bool] = False,
        # 设置时间步的间隔方式，默认为 "linspace"
        timestep_spacing: str = "linspace",
        # 设置时间步的偏移量，默认为 0
        steps_offset: int = 0,
        # 设置最终 sigma 的类型，可选，默认为 "zero"
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
        # 是否重新缩放 beta 以适应零 SNR，默认为 False
        rescale_betas_zero_snr: bool = False,
    ):
        # 检查是否提供了已训练的 beta 值
        if trained_betas is not None:
            # 将训练的 beta 值转换为 32 位浮点张量
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 根据线性调度生成 beta 值
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 根据缩放线性调度生成 beta 值，特定于潜在扩散模型
        elif beta_schedule == "scaled_linear":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 使用平方余弦调度生成 beta 值
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            # 如果调度未实现，抛出异常
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 如果需要重新缩放 beta 值，调用相应方法
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        # 计算 alphas 值
        self.alphas = 1.0 - self.betas
        # 计算 alphas 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 如果需要重新缩放，设置 alphas_cumprod 的最后一个值
        if rescale_betas_zero_snr:
            # 设置接近于 0 的值，以避免第一个 sigma 为无穷大
            self.alphas_cumprod[-1] = 2**-24

        # 目前仅支持 VP 类型噪声调度
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        # 计算 sigmas 值
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # 初始化噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 检查求解器类型是否合法
        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                # 如果求解器是某些类型，注册到配置中
                self.register_to_config(solver_type="bh2")
            else:
                # 如果求解器未实现，抛出异常
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        # 设置预测 x0 值
        self.predict_x0 = predict_x0
        # 可设置值的初始化
        self.num_inference_steps = None
        # 创建时间步的逆序数组
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        # 将时间步转换为张量
        self.timesteps = torch.from_numpy(timesteps)
        # 初始化模型输出列表
        self.model_outputs = [None] * solver_order
        # 初始化时间步列表
        self.timestep_list = [None] * solver_order
        # 初始化较低阶数目
        self.lower_order_nums = 0
        # 设置校正器禁用标志
        self.disable_corrector = disable_corrector
        # 设置求解器参数
        self.solver_p = solver_p
        # 初始化最后样本
        self.last_sample = None
        # 初始化步索引
        self._step_index = None
        # 初始化开始索引
        self._begin_index = None
        # 将 sigmas 移动到 CPU，减少 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度步骤后增加 1。
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过 `set_begin_index` 方法从管道设置。
        """
        return self._begin_index
    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制而来
    def set_begin_index(self, begin_index: int = 0):
        """
        设置调度器的起始索引。此函数应在推理之前从管道中运行。

        参数：
            begin_index (`int`):
                调度器的起始索引。
        """
        # 将给定的起始索引存储到对象的属性中
        self._begin_index = begin_index

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制而来
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        “动态阈值化：在每个采样步骤中，我们将 s 设置为 xt0（时间步 t 处 x_0 的预测）中的某个百分位绝对像素值，
        如果 s > 1，则将 xt0 阈值化到范围 [-s, s]，然后除以 s。动态阈值化推动饱和像素（那些接近 -1 和 1 的像素）向内移动，从而在每个步骤中积极防止像素饱和。我们发现，动态阈值化显著改善了照片真实感以及更好的图像-文本对齐，尤其是在使用非常大的引导权重时。”

        https://arxiv.org/abs/2205.11487
        """
        # 获取输入样本的类型
        dtype = sample.dtype
        # 获取样本的批量大小、通道数以及剩余维度
        batch_size, channels, *remaining_dims = sample.shape

        # 如果样本类型不是 float32 或 float64，则将其转换为 float 类型
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 为了进行分位数计算进行类型提升，且 clamp 在 CPU 的 half 类型上未实现

        # 将样本展平，以便在每个图像上进行分位数计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        # 计算样本的绝对值
        abs_sample = sample.abs()  # “某个百分位绝对像素值”

        # 计算每个样本的动态阈值 s
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 将 s 限制在 min=1 和 max=self.config.sample_max_value 之间
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 当限制为 min=1 时，相当于标准的 [-1, 1] 裁剪
        # 将 s 的形状调整为 (batch_size, 1)，以便在维度 0 上进行广播
        s = s.unsqueeze(1)  # (batch_size, 1) 因为 clamp 会在维度 0 上广播
        # 对样本进行阈值化，并将其除以 s
        sample = torch.clamp(sample, -s, s) / s  # “我们将 xt0 阈值化到范围 [-s, s]，然后除以 s”

        # 将样本的形状恢复到原始形状
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原始的数据类型
        sample = sample.to(dtype)

        # 返回处理后的样本
        return sample

    # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制而来
    # 定义私有方法 _sigma_to_t，接受 sigma 和 log_sigmas 作为参数
    def _sigma_to_t(self, sigma, log_sigmas):
        # 计算 sigma 的对数值，确保 sigma 不小于 1e-10，以避免取对数时出现负无穷
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # 计算 log_sigma 与 log_sigmas 之间的距离
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # 计算 sigmas 的范围，获取低索引和高索引
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        # 根据低索引和高索引获取 log_sigmas 的对应值
        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # 进行插值计算，得到权重 w
        w = (low - log_sigma) / (low - high)
        # 将权重限制在 [0, 1] 范围内
        w = np.clip(w, 0, 1)

        # 将插值转换为时间范围 t
        t = (1 - w) * low_idx + w * high_idx
        # 调整 t 的形状以匹配 sigma 的形状
        t = t.reshape(sigma.shape)
        # 返回计算得到的时间范围 t
        return t

    # 定义私有方法 _sigma_to_alpha_sigma_t，接受 sigma 作为参数
    # 从 DPMSolverMultistepScheduler 中复制而来
    def _sigma_to_alpha_sigma_t(self, sigma):
        # 计算 alpha_t，作为 sigma 的归一化因子
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        # 根据 alpha_t 计算 sigma_t
        sigma_t = sigma * alpha_t

        # 返回 alpha_t 和 sigma_t
        return alpha_t, sigma_t

    # 定义私有方法 _convert_to_karras，接受 in_sigmas 和 num_inference_steps 作为参数
    # 从 EulerDiscreteScheduler 中复制而来
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构造 Karras 等人 (2022) 的噪声调度。"""

        # 确保其他调度器在复制此函数时不会出现问题的黑客方式
        # TODO: 将此逻辑添加到其他调度器
        if hasattr(self.config, "sigma_min"):
            # 如果配置中有 sigma_min，使用其值
            sigma_min = self.config.sigma_min
        else:
            # 否则将 sigma_min 设置为 None
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            # 如果配置中有 sigma_max，使用其值
            sigma_max = self.config.sigma_max
        else:
            # 否则将 sigma_max 设置为 None
            sigma_max = None

        # 设置 sigma_min 和 sigma_max 的默认值
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        # 定义 rho 为 7.0，这是论文中使用的值
        rho = 7.0  
        # 创建从 0 到 1 的 ramp 数组，长度为 num_inference_steps
        ramp = np.linspace(0, 1, num_inference_steps)
        # 计算 sigma_min 和 sigma_max 的倒数
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 计算 sigmas，使用 ramp 进行插值
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回计算得到的 sigmas
        return sigmas

    # 定义 convert_model_output 方法，接受 model_output 和其他参数
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    # 该函数将模型输出转换为 UniPC 算法所需的相应类型
    ) -> torch.Tensor:
        r"""
        将模型输出转换为 UniPC 算法所需的对应类型。
    
        Args:
            model_output (`torch.Tensor`):
                学习的扩散模型的直接输出。
            timestep (`int`):
                扩散链中的当前离散时间步。
            sample (`torch.Tensor`):
                由扩散过程创建的当前样本实例。
    
        Returns:
            `torch.Tensor`:
                转换后的模型输出。
        """
        # 获取时间步，优先从位置参数 args 获取，如果 args 为空，则从关键字参数 kwargs 获取
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        # 如果样本为 None，则尝试从位置参数 args 获取第二个参数作为样本
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                # 如果样本仍然缺失，则抛出异常
                raise ValueError("missing `sample` as a required keyward argument")
        # 如果时间步不为空，警告用户该参数已被弃用
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )
    
        # 获取当前步骤索引对应的 sigma 值
        sigma = self.sigmas[self.step_index]
        # 将 sigma 转换为 alpha 和 sigma_t
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    
        # 如果预测 x0，则根据不同的预测类型计算 x0_pred
        if self.predict_x0:
            if self.config.prediction_type == "epsilon":
                # 计算 x0_pred
                x0_pred = (sample - sigma_t * model_output) / alpha_t
            elif self.config.prediction_type == "sample":
                # 直接使用模型输出作为 x0_pred
                x0_pred = model_output
            elif self.config.prediction_type == "v_prediction":
                # 根据 v_prediction 计算 x0_pred
                x0_pred = alpha_t * sample - sigma_t * model_output
            else:
                # 如果 prediction_type 不符合预期，则抛出异常
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )
    
            # 如果启用了阈值处理，则对 x0_pred 应用阈值
            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)
    
            # 返回计算后的 x0_pred
            return x0_pred
        else:
            # 否则根据不同的预测类型返回相应的输出
            if self.config.prediction_type == "epsilon":
                return model_output
            elif self.config.prediction_type == "sample":
                # 根据公式计算 epsilon
                epsilon = (sample - alpha_t * model_output) / sigma_t
                return epsilon
            elif self.config.prediction_type == "v_prediction":
                # 根据公式计算 epsilon
                epsilon = alpha_t * model_output + sigma_t * sample
                return epsilon
            else:
                # 如果 prediction_type 不符合预期，则抛出异常
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                    " `v_prediction` for the UniPCMultistepScheduler."
                )
    
    # 定义 multistep_uni_p_bh_update 函数
    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        *args,
        # 定义样本和顺序的可选参数
        sample: torch.Tensor = None,
        order: int = None,
        **kwargs,
    # 多步骤更新函数，进行模型输出的更新
    def multistep_uni_c_bh_update(
            self,
            this_model_output: torch.Tensor,  # 当前模型输出的张量
            *args,  # 可变参数，用于传递其他参数
            last_sample: torch.Tensor = None,  # 上一个样本，默认为 None
            this_sample: torch.Tensor = None,  # 当前样本，默认为 None
            order: int = None,  # 步骤顺序，默认为 None
            **kwargs,  # 关键字参数，用于传递其他参数
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep 复制而来
        def index_for_timestep(self, timestep, schedule_timesteps=None):  # 定义时间步的索引获取函数
            if schedule_timesteps is None:  # 如果未提供调度时间步
                schedule_timesteps = self.timesteps  # 使用对象的时间步
    
            index_candidates = (schedule_timesteps == timestep).nonzero()  # 找出与当前时间步匹配的索引候选
    
            if len(index_candidates) == 0:  # 如果没有匹配的索引候选
                step_index = len(self.timesteps) - 1  # 设定步索引为时间步的最后一个索引
            # 对于第一个“步骤”所取的 sigma 索引
            # 总是第二个索引（如果只有一个，则为最后一个）
            # 这样可以确保我们不会意外跳过 sigma
            # 如果我们从去噪调度的中间开始（例如图像到图像）
            elif len(index_candidates) > 1:  # 如果有多个匹配的候选
                step_index = index_candidates[1].item()  # 选择第二个索引
            else:  # 否则只有一个匹配
                step_index = index_candidates[0].item()  # 选择第一个索引
    
            return step_index  # 返回找到的步索引
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index 复制而来
        def _init_step_index(self, timestep):  # 初始化调度器的步索引计数器
            """
            Initialize the step_index counter for the scheduler.
            """
    
            if self.begin_index is None:  # 如果开始索引为空
                if isinstance(timestep, torch.Tensor):  # 如果时间步是张量
                    timestep = timestep.to(self.timesteps.device)  # 将其转换到时间步的设备
                self._step_index = self.index_for_timestep(timestep)  # 获取并设置当前步索引
            else:  # 如果开始索引不为空
                self._step_index = self._begin_index  # 使用提供的开始索引
    
        def step(  # 定义步骤函数
            self,
            model_output: torch.Tensor,  # 模型输出的张量
            timestep: Union[int, torch.Tensor],  # 当前时间步，可以是整数或张量
            sample: torch.Tensor,  # 当前样本的张量
            return_dict: bool = True,  # 是否返回字典形式的结果，默认为 True
    # 定义函数，返回类型为调度器输出或元组
    ) -> Union[SchedulerOutput, Tuple]:
        """
        通过反向 SDE 从前一个时间步预测样本。该函数使用多步 UniPC 传播样本。
    
        参数：
            model_output (`torch.Tensor`):
                从学习的扩散模型直接输出的张量。
            timestep (`int`):
                当前在扩散链中的离散时间步。
            sample (`torch.Tensor`):
                通过扩散过程创建的当前样本实例。
            return_dict (`bool`):
                是否返回 [`~schedulers.scheduling_utils.SchedulerOutput`] 或元组。
    
        返回：
            [`~schedulers.scheduling_utils.SchedulerOutput`] 或元组:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_utils.SchedulerOutput`]，否则返回元组，
                其中第一个元素是样本张量。
    
        """
        # 检查推理步骤是否为 None，若是则引发错误
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
    
        # 如果步骤索引为 None，则初始化步骤索引
        if self.step_index is None:
            self._init_step_index(timestep)
    
        # 检查是否使用修正器，条件是步骤索引大于 0，并且前一个步骤没有被禁用且上一个样本不为 None
        use_corrector = (
            self.step_index > 0 and self.step_index - 1 not in self.disable_corrector and self.last_sample is not None
        )
    
        # 转换模型输出以便后续使用
        model_output_convert = self.convert_model_output(model_output, sample=sample)
        # 如果使用修正器，更新样本
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )
    
        # 更新模型输出和时间步列表
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
    
        # 将当前模型输出和时间步存入最后的位置
        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep
    
        # 根据配置决定当前阶数
        if self.config.lower_order_final:
            this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
        else:
            this_order = self.config.solver_order
    
        # 设置当前阶数并进行多步的热身
        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0  # 确保当前阶数大于 0
    
        # 更新最后的样本
        self.last_sample = sample
        # 更新前一个样本
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # 传递原始未转换的模型输出，以防使用 solver-p
            sample=sample,
            order=self.this_order,
        )
    
        # 如果低阶数量小于配置的解算器阶数，则增加低阶数量
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1
    
        # 完成后将步骤索引增加一
        self._step_index += 1
    
        # 如果不返回字典，则返回前一个样本作为元组
        if not return_dict:
            return (prev_sample,)
    
        # 返回调度器输出对象，包含前一个样本
        return SchedulerOutput(prev_sample=prev_sample)
    # 定义一个方法，用于缩放模型输入，接受一个张量样本及可变参数
    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器的互换性。

        Args:
            sample (`torch.Tensor`):
                输入样本。

        Returns:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 直接返回输入样本，不做任何处理
        return sample

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.add_noise 拷贝
    def add_noise(
        self,
        original_samples: torch.Tensor,  # 原始样本张量
        noise: torch.Tensor,              # 噪声张量
        timesteps: torch.IntTensor,      # 时间步张量
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 的设备和数据类型与 original_samples 相同
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备类型，如果是 MPS 并且时间步是浮点数
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # MPS 不支持 float64，因此将时间步转换为 float32
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将调度时间步转换为与原始样本相同的设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # 如果 begin_index 为 None，表示调度器用于训练或管道未实现 set_begin_index
        if self.begin_index is None:
            # 计算每个时间步对应的步骤索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise 在第一次去噪步骤之后调用（用于修复）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add_noise 在第一次去噪步骤之前调用以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 根据步骤索引提取 sigma，并展平
        sigma = sigmas[step_indices].flatten()
        # 如果 sigma 的维度小于原始样本的维度，则增加维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 将 sigma 转换为 alpha_t 和 sigma_t
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # 根据 alpha_t 和 sigma_t 生成带噪声的样本
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        # 返回带噪声的样本
        return noisy_samples

    # 返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```