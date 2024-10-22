# `.\diffusers\schedulers\scheduling_deis_multistep.py`

```py
# Copyright 2024 FLAIR Lab and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 你可以在遵循许可证的情况下使用此文件。
# 可以通过以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面同意，软件在“按原样”基础上分发，
# 不提供任何形式的明示或暗示的保证或条件。
# 有关许可证的具体权限和限制，请参见许可证文档。

# 声明：请查看 https://arxiv.org/abs/2204.13902 和 https://github.com/qsh-zh/deis 以获取更多信息
# 此代码库是基于 https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py 进行修改的

import math  # 导入数学模块，用于数学计算
from typing import List, Optional, Tuple, Union  # 导入类型提示相关的类型

import numpy as np  # 导入 NumPy 库，用于数组和数学运算
import torch  # 导入 PyTorch 库，用于张量操作和深度学习

from ..configuration_utils import ConfigMixin, register_to_config  # 从配置工具导入混合类和注册函数
from ..utils import deprecate  # 从工具模块导入弃用标记函数
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput  # 从调度工具模块导入调度相关的类

# 从 diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar 复制的函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 传入参数：扩散时间步数
    max_beta=0.999,  # 传入参数：使用的最大 beta 值，默认值为 0.999
    alpha_transform_type="cosine",  # 传入参数：alpha 转换类型，默认为 "cosine"
):
    """
    创建一个 beta 调度，以离散化给定的 alpha_t_bar 函数，该函数定义了时间 t = [0,1] 上
    (1-beta) 的累积乘积。

    包含一个 alpha_bar 函数，该函数接受参数 t 并将其转换为扩散过程中该部分的 (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 生成的 beta 数量。
        max_beta (`float`): 使用的最大 beta 值；使用小于 1 的值以防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认值为 `cosine`): alpha_bar 的噪声调度类型。
                     可选择 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于更新模型输出的 beta 值
    """
    if alpha_transform_type == "cosine":  # 如果选择的转换类型为 "cosine"
        def alpha_bar_fn(t):  # 定义 alpha_bar 函数
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2  # 计算并返回余弦值的平方

    elif alpha_transform_type == "exp":  # 如果选择的转换类型为 "exp"
        def alpha_bar_fn(t):  # 定义 alpha_bar 函数
            return math.exp(t * -12.0)  # 计算并返回指数衰减值

    else:  # 如果选择的转换类型不支持
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")  # 抛出值错误

    betas = []  # 初始化一个空列表用于存储 beta 值
    for i in range(num_diffusion_timesteps):  # 遍历每个扩散时间步
        t1 = i / num_diffusion_timesteps  # 计算当前时间步 t1
        t2 = (i + 1) / num_diffusion_timesteps  # 计算下一个时间步 t2
        # 计算 beta 值并添加到列表中，限制最大值为 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)  # 将 beta 列表转换为 PyTorch 张量并返回


class DEISMultistepScheduler(SchedulerMixin, ConfigMixin):  # 定义 DEISMultistepScheduler 类，继承自调度器和配置混合类
    """
    `DEISMultistepScheduler` 是一个快速高阶解算器，用于扩散常微分方程（ODEs）。
    # 该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查阅父类文档以获取库为所有调度程序实现的通用方法，例如加载和保存。
    
    # 参数说明：
    # num_train_timesteps (`int`, defaults to 1000):
    #     用于训练模型的扩散步骤数量。
    # beta_start (`float`, defaults to 0.0001):
    #     推断的起始 `beta` 值。
    # beta_end (`float`, defaults to 0.02):
    #     最终的 `beta` 值。
    # beta_schedule (`str`, defaults to `"linear"`):
    #     beta 计划，从 beta 范围映射到一系列用于模型步骤的 betas。可选择 `linear`、`scaled_linear` 或 `squaredcos_cap_v2`。
    # trained_betas (`np.ndarray`, *optional*):
    #     直接传递 beta 数组给构造函数，以绕过 `beta_start` 和 `beta_end`。
    # solver_order (`int`, defaults to 2):
    #     DEIS 顺序，可以是 `1`、`2` 或 `3`。建议使用 `solver_order=2` 进行引导采样，使用 `solver_order=3` 进行无条件采样。
    # prediction_type (`str`, defaults to `epsilon`):
    #     调度程序函数的预测类型；可以是 `epsilon`（预测扩散过程的噪声）、`sample`（直接预测噪声样本）或 `v_prediction`（见 [Imagen Video](https://imagen.research.google/video/paper.pdf) 论文第 2.4 节）。
    # thresholding (`bool`, defaults to `False`):
    #     是否使用“动态阈值”方法。这对于如稳定扩散的潜空间扩散模型不适用。
    # dynamic_thresholding_ratio (`float`, defaults to 0.995):
    #     动态阈值方法的比率。仅在 `thresholding=True` 时有效。
    # sample_max_value (`float`, defaults to 1.0):
    #     动态阈值的阈值值。仅在 `thresholding=True` 时有效。
    # algorithm_type (`str`, defaults to `deis`):
    #     求解器的算法类型。
    # lower_order_final (`bool`, defaults to `True`):
    #     是否在最后步骤中使用低阶求解器。仅在推理步骤小于 15 时有效。
    # use_karras_sigmas (`bool`, *optional*, defaults to `False`):
    #     是否在采样过程中使用 Karras sigmas 作为噪声计划中的步长。如果为 `True`，则 sigmas 根据噪声水平序列 {σi} 确定。
    # timestep_spacing (`str`, defaults to `"linspace"`):
    #     时间步的缩放方式。有关更多信息，请参考 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) 的表 2。
    # steps_offset (`int`, defaults to 0):
    #     添加到推理步骤的偏移量，根据某些模型系列的要求。
    # 创建一个包含所有 KarrasDiffusionSchedulers 名称的列表
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置默认的求解器阶数为 1
    order = 1

    # 注册到配置中，定义初始化函数
    @register_to_config
    def __init__(
        # 设置训练时间步的数量，默认值为 1000
        num_train_timesteps: int = 1000,
        # 设置 beta 的起始值，默认值为 0.0001
        beta_start: float = 0.0001,
        # 设置 beta 的结束值，默认值为 0.02
        beta_end: float = 0.02,
        # 设置 beta 的调度方式，默认值为 "linear"
        beta_schedule: str = "linear",
        # 可选参数，设置训练的 beta 数组，默认值为 None
        trained_betas: Optional[np.ndarray] = None,
        # 设置求解器的阶数，默认值为 2
        solver_order: int = 2,
        # 设置预测类型，默认值为 "epsilon"
        prediction_type: str = "epsilon",
        # 设置是否使用阈值处理，默认值为 False
        thresholding: bool = False,
        # 设置动态阈值比例，默认值为 0.995
        dynamic_thresholding_ratio: float = 0.995,
        # 设置样本的最大值，默认值为 1.0
        sample_max_value: float = 1.0,
        # 设置算法类型，默认值为 "deis"
        algorithm_type: str = "deis",
        # 设置求解器类型，默认值为 "logrho"
        solver_type: str = "logrho",
        # 设置是否在最后阶段使用较低的阶数，默认值为 True
        lower_order_final: bool = True,
        # 可选参数，设置是否使用 Karras sigma，默认值为 False
        use_karras_sigmas: Optional[bool] = False,
        # 设置时间步的间距类型，默认值为 "linspace"
        timestep_spacing: str = "linspace",
        # 设置步数偏移，默认值为 0
        steps_offset: int = 0,
    ):
        # 检查已训练的 beta 值是否为 None
        if trained_betas is not None:
            # 将训练的 beta 值转换为浮点型张量
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 检查 beta 调度类型是否为线性
        elif beta_schedule == "linear":
            # 生成从 beta_start 到 beta_end 的线性序列
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 检查 beta 调度类型是否为缩放线性
        elif beta_schedule == "scaled_linear":
            # 该调度特定于潜在扩散模型
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 检查 beta 调度类型是否为平方余弦 cap v2
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            # 如果不支持的调度类型，抛出未实现错误
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 计算 alphas，等于 1 减去 betas
        self.alphas = 1.0 - self.betas
        # 计算 alphas 的累积乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 当前仅支持 VP 类型噪声调度
        self.alpha_t = torch.sqrt(self.alphas_cumprod)
        # 计算 sigma_t，等于 1 减去 alphas_cumprod 的平方根
        self.sigma_t = torch.sqrt(1 - self.alphas_cumprod)
        # 计算 lambda_t，等于 alpha_t 和 sigma_t 的对数差
        self.lambda_t = torch.log(self.alpha_t) - torch.log(self.sigma_t)
        # 计算 sigmas，等于 (1 - alphas_cumprod) 除以 alphas_cumprod 的平方根
        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5

        # 设置初始噪声分布的标准差
        self.init_noise_sigma = 1.0

        # DEIS 设置
        if algorithm_type not in ["deis"]:
            # 如果算法类型是 dpmsolver 或 dpmsolver++
            if algorithm_type in ["dpmsolver", "dpmsolver++"]:
                # 注册算法类型到配置
                self.register_to_config(algorithm_type="deis")
            else:
                # 抛出未实现错误
                raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")

        # 检查求解器类型是否为 logrho
        if solver_type not in ["logrho"]:
            # 如果求解器类型是 midpoint, heun, bh1, bh2
            if solver_type in ["midpoint", "heun", "bh1", "bh2"]:
                # 注册求解器类型到配置
                self.register_to_config(solver_type="logrho")
            else:
                # 抛出未实现错误
                raise NotImplementedError(f"solver type {solver_type} is not implemented for {self.__class__}")

        # 可设置的值
        self.num_inference_steps = None
        # 生成从 0 到 num_train_timesteps - 1 的时间步，反转顺序
        timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype=np.float32)[::-1].copy()
        # 将时间步转换为张量
        self.timesteps = torch.from_numpy(timesteps)
        # 初始化模型输出列表，长度为 solver_order
        self.model_outputs = [None] * solver_order
        # 记录低阶数
        self.lower_order_nums = 0
        # 初始化步索引
        self._step_index = None
        # 初始化开始索引
        self._begin_index = None
        # 将 sigmas 移到 CPU，以避免过多的 CPU/GPU 通信
        self.sigmas = self.sigmas.to("cpu")  # to avoid too much CPU/GPU communication

    @property
    def step_index(self):
        """
        当前时间步的索引计数器。每次调度器步骤后增加 1。
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        第一个时间步的索引。应通过 `set_begin_index` 方法从管道中设置。
        """
        return self._begin_index

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index 复制
    # 设置调度器的起始索引，默认值为0
    def set_begin_index(self, begin_index: int = 0):
        # 文档字符串，说明函数的用途和参数
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.
    
        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        # 将传入的起始索引值存储到实例变量中
        self._begin_index = begin_index
    
        # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制的函数
        def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
            # 文档字符串，描述动态阈值处理的原理和效果
            """
            "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
            prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
            s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
            pixels from saturation at each step. We find that dynamic thresholding results in significantly better
            photorealism as well as better image-text alignment, especially when using very large guidance weights."
    
            https://arxiv.org/abs/2205.11487
            """
            # 获取输入样本的数值类型
            dtype = sample.dtype
            # 获取样本的批次大小、通道数及剩余维度
            batch_size, channels, *remaining_dims = sample.shape
    
            # 检查数据类型，如果不是浮点数，则转换为浮点数
            if dtype not in (torch.float32, torch.float64):
                sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half
    
            # 将样本扁平化以进行量化计算
            sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
    
            # 计算样本的绝对值
            abs_sample = sample.abs()  # "a certain percentile absolute pixel value"
    
            # 计算每个图像的动态阈值
            s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
            # 限制阈值在指定范围内
            s = torch.clamp(
                s, min=1, max=self.config.sample_max_value
            )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
            # 扩展维度以适应广播
            s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
            # 将样本限制在[-s, s]范围内并归一化
            sample = torch.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"
    
            # 恢复样本的原始形状
            sample = sample.reshape(batch_size, channels, *remaining_dims)
            # 将样本转换回原始数据类型
            sample = sample.to(dtype)
    
            # 返回处理后的样本
            return sample
    
        # 从 diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t 复制的函数
        def _sigma_to_t(self, sigma, log_sigmas):
            # 计算对数sigma值，确保不小于1e-10
            log_sigma = np.log(np.maximum(sigma, 1e-10))
    
            # 计算对数sigma的分布
            dists = log_sigma - log_sigmas[:, np.newaxis]
    
            # 找到sigma的范围
            low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
    
            # 获取低和高的对数sigma值
            low = log_sigmas[low_idx]
            high = log_sigmas[high_idx]
    
            # 进行sigma的插值
            w = (low - log_sigma) / (low - high)
            w = np.clip(w, 0, 1)
    
            # 将插值转换为时间范围
            t = (1 - w) * low_idx + w * high_idx
            # 重新调整形状以匹配sigma的形状
            t = t.reshape(sigma.shape)
            # 返回时间值
            return t
    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep 导入的函数，用于将 sigma 转换为 alpha 和 sigma_t
    def _sigma_to_alpha_sigma_t(self, sigma):
        # 计算 alpha_t，公式为 1 / sqrt(sigma^2 + 1)
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        # 计算 sigma_t，公式为 sigma * alpha_t
        sigma_t = sigma * alpha_t

        # 返回计算得到的 alpha_t 和 sigma_t
        return alpha_t, sigma_t

    # 从 diffusers.schedulers.scheduling_euler_discrete 导入的函数，用于将输入 sigma 转换为 Karras 的格式
    def _convert_to_karras(self, in_sigmas: torch.Tensor, num_inference_steps) -> torch.Tensor:
        """构建 Karras 等人 (2022) 的噪声调度。"""

        # 确保其他调度器复制此函数时不会出错的黑客方案
        # TODO: 将此逻辑添加到其他调度器中
        if hasattr(self.config, "sigma_min"):
            # 获取 sigma_min，如果配置中存在
            sigma_min = self.config.sigma_min
        else:
            # 如果配置中不存在，设置为 None
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            # 获取 sigma_max，如果配置中存在
            sigma_max = self.config.sigma_max
        else:
            # 如果配置中不存在，设置为 None
            sigma_max = None

        # 设置 sigma_min 为输入 sigmas 的最后一个值，如果它是 None
        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        # 设置 sigma_max 为输入 sigmas 的第一个值，如果它是 None
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        # 定义 rho 的值为 7.0，引用文献中使用的值
        rho = 7.0  # 7.0 is the value used in the paper
        # 生成从 0 到 1 的 ramp 数组，长度为 num_inference_steps
        ramp = np.linspace(0, 1, num_inference_steps)
        # 计算 min_inv_rho 和 max_inv_rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        # 根据公式生成 sigmas 数组
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        # 返回生成的 sigmas
        return sigmas

    # 定义 convert_model_output 函数，用于处理模型输出
    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        将模型输出转换为 DEIS 算法所需的对应类型。

        参数:
            model_output (`torch.Tensor`):
                来自学习的扩散模型的直接输出。
            timestep (`int`):
                当前扩散链中的离散时间步。
            sample (`torch.Tensor`):
                扩散过程中创建的当前样本实例。

        返回:
            `torch.Tensor`:
                转换后的模型输出。
        """
        # 从 args 中提取 timestep，如果没有则从 kwargs 中提取，默认为 None
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)
        # 如果 sample 为 None，尝试从 args 中提取
        if sample is None:
            if len(args) > 1:
                sample = args[1]
            else:
                # 如果没有提供 sample，则抛出错误
                raise ValueError("missing `sample` as a required keyward argument")
        # 如果 timestep 不是 None，发出弃用警告
        if timestep is not None:
            deprecate(
                "timesteps",
                "1.0.0",
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
            )

        # 获取当前步的 sigma 值
        sigma = self.sigmas[self.step_index]
        # 将 sigma 转换为 alpha_t 和 sigma_t
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # 根据配置类型进行不同的模型输出处理
        if self.config.prediction_type == "epsilon":
            # 计算基于 epsilon 的预测
            x0_pred = (sample - sigma_t * model_output) / alpha_t
        elif self.config.prediction_type == "sample":
            # 直接将模型输出作为预测
            x0_pred = model_output
        elif self.config.prediction_type == "v_prediction":
            # 计算基于 v 的预测
            x0_pred = alpha_t * sample - sigma_t * model_output
        else:
            # 如果 prediction_type 不符合要求，抛出错误
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction` for the DEISMultistepScheduler."
            )

        # 如果开启阈值处理，则对预测值进行阈值处理
        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)

        # 如果算法类型为 deis，返回转换后的样本
        if self.config.algorithm_type == "deis":
            return (sample - alpha_t * x0_pred) / sigma_t
        else:
            # 抛出未实现错误，表明仅支持 log-rho multistep deis
            raise NotImplementedError("only support log-rho multistep deis now")

    # 定义 deis_first_order_update 函数，接受模型输出和可变参数
    def deis_first_order_update(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:  # 定义函数返回类型为 torch.Tensor
        """  # 开始函数的文档字符串
        One step for the first-order DEIS (equivalent to DDIM).  # 描述该函数为一阶 DEIS 步骤（等同于 DDIM）

        Args:  # 参数说明开始
            model_output (`torch.Tensor`):  # 参数 model_output，类型为 torch.Tensor
                The direct output from the learned diffusion model.  # 描述为从学习到的扩散模型获得的直接输出
            timestep (`int`):  # 参数 timestep，类型为 int
                The current discrete timestep in the diffusion chain.  # 描述为扩散链中的当前离散时间步
            prev_timestep (`int`):  # 参数 prev_timestep，类型为 int
                The previous discrete timestep in the diffusion chain.  # 描述为扩散链中的前一个离散时间步
            sample (`torch.Tensor`):  # 参数 sample，类型为 torch.Tensor
                A current instance of a sample created by the diffusion process.  # 描述为扩散过程创建的当前样本实例

        Returns:  # 返回值说明开始
            `torch.Tensor`:  # 返回类型为 torch.Tensor
                The sample tensor at the previous timestep.  # 描述为在前一个时间步的样本张量
        """  # 结束函数的文档字符串
        timestep = args[0] if len(args) > 0 else kwargs.pop("timestep", None)  # 获取当前时间步，如果没有则从关键字参数中提取
        prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)  # 获取前一个时间步，如果没有则从关键字参数中提取
        if sample is None:  # 检查 sample 是否为 None
            if len(args) > 2:  # 如果 args 的长度大于 2
                sample = args[2]  # 从 args 中获取 sample
            else:  # 否则
                raise ValueError(" missing `sample` as a required keyward argument")  # 抛出缺少 sample 的异常
        if timestep is not None:  # 如果当前时间步不为 None
            deprecate(  # 调用 deprecate 函数以发出弃用警告
                "timesteps",  # 被弃用的参数名称
                "1.0.0",  # 版本号
                "Passing `timesteps` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",  # 弃用说明
            )

        if prev_timestep is not None:  # 如果前一个时间步不为 None
            deprecate(  # 调用 deprecate 函数以发出弃用警告
                "prev_timestep",  # 被弃用的参数名称
                "1.0.0",  # 版本号
                "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",  # 弃用说明
            )

        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]  # 获取当前和前一个时间步的 sigma 值
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)  # 将 sigma_t 转换为 alpha_t 和 sigma_t
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)  # 将 sigma_s 转换为 alpha_s 和 sigma_s
        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)  # 计算 lambda_t 为 alpha_t 和 sigma_t 的对数差
        lambda_s = torch.log(alpha_s) - torch.log(sigma_s)  # 计算 lambda_s 为 alpha_s 和 sigma_s 的对数差

        h = lambda_t - lambda_s  # 计算 h 为 lambda_t 和 lambda_s 的差
        if self.config.algorithm_type == "deis":  # 检查算法类型是否为 "deis"
            x_t = (alpha_t / alpha_s) * sample - (sigma_t * (torch.exp(h) - 1.0)) * model_output  # 计算当前样本 x_t
        else:  # 否则
            raise NotImplementedError("only support log-rho multistep deis now")  # 抛出不支持的算法类型异常
        return x_t  # 返回计算得到的样本 x_t

    def multistep_deis_second_order_update(  # 定义 multistep_deis_second_order_update 函数
        self,  # 类实例
        model_output_list: List[torch.Tensor],  # 参数 model_output_list，类型为 torch.Tensor 的列表
        *args,  # 可变位置参数
        sample: torch.Tensor = None,  # 参数 sample，默认为 None
        **kwargs,  # 可变关键字参数
    # 定义一个函数，返回类型为 torch.Tensor
        ) -> torch.Tensor:
            """
            第二阶多步 DEIS 的一步计算。
    
            参数：
                model_output_list (`List[torch.Tensor]`):
                    当前和后续时间步的学习扩散模型直接输出。
                sample (`torch.Tensor`):
                    扩散过程生成的当前样本实例。
    
            返回：
                `torch.Tensor`:
                    上一时间步的样本张量。
            """
            # 获取时间步列表，如果没有则从关键字参数中获取
            timestep_list = args[0] if len(args) > 0 else kwargs.pop("timestep_list", None)
            # 获取前一个时间步，如果没有则从关键字参数中获取
            prev_timestep = args[1] if len(args) > 1 else kwargs.pop("prev_timestep", None)
            # 如果样本为 None，则尝试从参数中获取样本
            if sample is None:
                if len(args) > 2:
                    sample = args[2]
                else:
                    # 如果样本仍然为 None，则引发错误
                    raise ValueError(" missing `sample` as a required keyward argument")
            # 如果时间步列表不为 None，则发出弃用警告
            if timestep_list is not None:
                deprecate(
                    "timestep_list",
                    "1.0.0",
                    "Passing `timestep_list` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
                )
    
            # 如果前一个时间步不为 None，则发出弃用警告
            if prev_timestep is not None:
                deprecate(
                    "prev_timestep",
                    "1.0.0",
                    "Passing `prev_timestep` is deprecated and has no effect as model output conversion is now handled via an internal counter `self.step_index`",
                )
    
            # 获取当前和前后时间步的 sigma 值
            sigma_t, sigma_s0, sigma_s1 = (
                self.sigmas[self.step_index + 1],
                self.sigmas[self.step_index],
                self.sigmas[self.step_index - 1],
            )
    
            # 将 sigma 转换为 alpha 和 sigma_t
            alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
            alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
            alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
    
            # 获取最后两个模型输出
            m0, m1 = model_output_list[-1], model_output_list[-2]
    
            # 计算 rho 值
            rho_t, rho_s0, rho_s1 = sigma_t / alpha_t, sigma_s0 / alpha_s0, sigma_s1 / alpha_s1
    
            # 检查算法类型是否为 "deis"
            if self.config.algorithm_type == "deis":
    
                # 定义积分函数
                def ind_fn(t, b, c):
                    # Integrate[(log(t) - log(c)) / (log(b) - log(c)), {t}]
                    return t * (-np.log(c) + np.log(t) - 1) / (np.log(b) - np.log(c))
    
                # 计算系数
                coef1 = ind_fn(rho_t, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s0, rho_s1)
                coef2 = ind_fn(rho_t, rho_s1, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s0)
    
                # 计算 x_t
                x_t = alpha_t * (sample / alpha_s0 + coef1 * m0 + coef2 * m1)
                # 返回计算结果
                return x_t
            else:
                # 如果算法类型不支持，则引发未实现的错误
                raise NotImplementedError("only support log-rho multistep deis now")
    
        # 定义一个多步 DEIS 第三阶更新的函数
        def multistep_deis_third_order_update(
            self,
            model_output_list: List[torch.Tensor],
            *args,
            # 当前样本实例，默认为 None
            sample: torch.Tensor = None,
            **kwargs,
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep 复制
    # 根据时间步初始化索引
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        # 如果未提供时间调度步，则使用默认时间步
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps
    
        # 找到与当前时间步匹配的候选索引
        index_candidates = (schedule_timesteps == timestep).nonzero()
    
        # 如果没有找到匹配的候选索引
        if len(index_candidates) == 0:
            # 将步骤索引设置为时间步的最后一个索引
            step_index = len(self.timesteps) - 1
        # 如果找到多个候选索引
        # 第一个步骤的 sigma 索引总是第二个索引（如果只有一个则是最后一个）
        # 这样可以确保在去噪调度中不会意外跳过 sigma
        elif len(index_candidates) > 1:
            # 使用第二个候选索引作为步骤索引
            step_index = index_candidates[1].item()
        else:
            # 否则，使用第一个候选索引作为步骤索引
            step_index = index_candidates[0].item()
    
        # 返回最终步骤索引
        return step_index
    
        # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index 中复制
        def _init_step_index(self, timestep):
            """
            初始化调度器的步骤索引计数器。
            """
    
            # 如果开始索引为 None
            if self.begin_index is None:
                # 如果时间步是张量类型，则将其转移到相应设备
                if isinstance(timestep, torch.Tensor):
                    timestep = timestep.to(self.timesteps.device)
                # 使用 index_for_timestep 方法初始化步骤索引
                self._step_index = self.index_for_timestep(timestep)
            else:
                # 否则使用预设的开始索引
                self._step_index = self._begin_index
    
        # 执行一步计算
        def step(
            self,
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
            return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        从前一个时间步预测样本，通过反转 SDE。此函数使用多步 DEIS 传播样本。

        参数：
            model_output (`torch.Tensor`):
                从学习的扩散模型直接输出的张量。
            timestep (`int`):
                扩散链中当前离散时间步。
            sample (`torch.Tensor`):
                通过扩散过程创建的当前样本实例。
            return_dict (`bool`):
                是否返回 [`~schedulers.scheduling_utils.SchedulerOutput`] 或 `tuple`。

        返回：
            [`~schedulers.scheduling_utils.SchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，则返回 [`~schedulers.scheduling_utils.SchedulerOutput`]，否则返回一个元组，
                其中第一个元素是样本张量。

        """
        # 检查推理步骤数量是否为 None，若是则抛出异常
        if self.num_inference_steps is None:
            raise ValueError(
                "推理步骤数量为 'None'，您需要在创建调度器后运行 'set_timesteps'"
            )

        # 检查当前步骤索引是否为 None，若是则初始化步骤索引
        if self.step_index is None:
            self._init_step_index(timestep)

        # 判断是否为较低阶最终更新的条件
        lower_order_final = (
            (self.step_index == len(self.timesteps) - 1) and self.config.lower_order_final and len(self.timesteps) < 15
        )
        # 判断是否为较低阶第二更新的条件
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        # 转换模型输出为适合当前样本的格式
        model_output = self.convert_model_output(model_output, sample=sample)
        # 更新模型输出缓存，将当前模型输出存储到最后一个位置
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        # 根据配置选择合适的更新方法
        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            # 使用一阶更新方法计算前一个样本
            prev_sample = self.deis_first_order_update(model_output, sample=sample)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            # 使用二阶更新方法计算前一个样本
            prev_sample = self.multistep_deis_second_order_update(self.model_outputs, sample=sample)
        else:
            # 使用三阶更新方法计算前一个样本
            prev_sample = self.multistep_deis_third_order_update(self.model_outputs, sample=sample)

        # 更新较低阶次数计数器
        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # 完成后将步骤索引加一
        self._step_index += 1

        # 如果不返回字典，则返回包含前一个样本的元组
        if not return_dict:
            return (prev_sample,)

        # 返回前一个样本的调度输出
        return SchedulerOutput(prev_sample=prev_sample)
    # 定义一个方法，用于根据当前时间步缩放去噪模型输入
    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器之间的互换性。

        Args:
            sample (`torch.Tensor`):
                输入样本。

        Returns:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 返回未修改的输入样本
        return sample

    # 从 diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.add_noise 复制的代码
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # 确保 sigmas 和 timesteps 与 original_samples 具有相同的设备和数据类型
        sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
        # 检查设备类型，如果是 MPS 且 timesteps 是浮点型
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # MPS 不支持 float64 数据类型
            schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
        else:
            # 将调度时间步转换为与原始样本相同的设备
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # 如果 begin_index 为 None，表示调度器用于训练或管道未实现 set_begin_index
        if self.begin_index is None:
            # 根据时间步获取步索引
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # 在第一次去噪步骤后调用 add_noise（用于修补）
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # 在第一次去噪步骤之前调用 add_noise 以创建初始潜在图像（img2img）
            step_indices = [self.begin_index] * timesteps.shape[0]

        # 根据步索引获取 sigma，并将其扁平化
        sigma = sigmas[step_indices].flatten()
        # 如果 sigma 的形状小于原始样本的形状，则在最后一个维度添加一个维度
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        # 将 sigma 转换为 alpha_t 和 sigma_t
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        # 生成带噪声的样本
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        # 返回带噪声的样本
        return noisy_samples

    # 定义方法以返回训练时间步的数量
    def __len__(self):
        return self.config.num_train_timesteps
```