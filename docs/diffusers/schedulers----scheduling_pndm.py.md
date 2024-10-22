# `.\diffusers\schedulers\scheduling_pndm.py`

```py
# 版权信息，声明版权所有者和使用条款
# Copyright 2024 Zhejiang University Team and The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版（“许可证”）授权；在遵循许可证的前提下使用此文件。
# 您可以在以下网址获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件按“原样”分发，没有任何形式的保证或条件，
# 无论是明示还是暗示。
# 请参阅许可证以获取有关权限和限制的具体说明。

# 声明：该文件受到 https://github.com/ermongroup/ddim 的强烈影响

# 导入数学模块
import math
# 从类型提示导入必要的类型
from typing import List, Optional, Tuple, Union

# 导入 numpy 和 torch 库
import numpy as np
import torch

# 从配置工具导入所需的混合类和注册函数
from ..configuration_utils import ConfigMixin, register_to_config
# 从调度工具导入调度器和输出类
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput

# 定义生成 beta 调度的函数，基于 alpha_t_bar 函数
def betas_for_alpha_bar(
    num_diffusion_timesteps,  # 生成的 beta 数量
    max_beta=0.999,  # 最大 beta 值
    alpha_transform_type="cosine",  # alpha 转换类型
):
    """
    创建一个 beta 调度，离散化给定的 alpha_t_bar 函数，定义了时间 t=[0,1] 上 (1-beta) 的累积乘积。

    包含一个 alpha_bar 函数，接受 t 参数并将其转换为扩散过程中的 (1-beta) 的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 要生成的 beta 数量。
        max_beta (`float`): 要使用的最大 beta 值；使用小于 1 的值以
                     防止奇点。
        alpha_transform_type (`str`, *可选*，默认为 `cosine`): alpha_bar 的噪声调度类型。
                     选择 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于更新模型输出的 betas
    """
    # 根据选择的 alpha 转换类型定义 alpha_bar 函数
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):  # 定义基于余弦的 alpha_bar 函数
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):  # 定义基于指数的 alpha_bar 函数
            return math.exp(t * -12.0)

    else:
        # 如果 alpha 转换类型不被支持，抛出错误
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []  # 初始化空列表以存储 beta 值
    # 遍历每个扩散时间步，计算相应的 beta 值
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps  # 当前时间点
        t2 = (i + 1) / num_diffusion_timesteps  # 下一个时间点
        # 计算 beta 值并添加到列表中
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 返回作为 PyTorch 张量的 beta 值
    return torch.tensor(betas, dtype=torch.float32)

# 定义 PNDMScheduler 类，使用伪数值方法进行扩散模型调度
class PNDMScheduler(SchedulerMixin, ConfigMixin):
    """
    `PNDMScheduler` 使用伪数值方法进行扩散模型的调度，如龙格-库塔和线性多步方法。

    此模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。有关所有调度器的通用方法的库文档，请检查超类文档，如加载和保存。
    # 参数说明
    Args:
        num_train_timesteps (`int`, defaults to 1000):
            # 模型训练的扩散步骤数量，默认为 1000
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            # 推理的起始 `beta` 值，默认为 0.0001
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            # 最终的 `beta` 值，默认为 0.02
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            # beta 调度策略，从 beta 范围到模型步进的 beta 序列的映射。可选值包括 `linear`、`scaled_linear` 或 `squaredcos_cap_v2`
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            # 直接将 beta 数组传递给构造函数，以绕过 `beta_start` 和 `beta_end`
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        skip_prk_steps (`bool`, defaults to `False`):
            # 允许调度器跳过原始论文中定义的 Runge-Kutta 步骤，这些步骤在 PLMS 步骤之前是必需的
            Allows the scheduler to skip the Runge-Kutta steps defined in the original paper as being required before
            PLMS steps.
        set_alpha_to_one (`bool`, defaults to `False`):
            # 每个扩散步骤使用该步骤和前一步的 alpha 乘积值。对于最后一步没有前一个 alpha。当选项为 `True` 时，前一个 alpha 乘积固定为 1， 否则使用第 0 步的 alpha 值
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to `1`,
            otherwise it uses the alpha value at step 0.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            # 调度函数的预测类型；可以是 `epsilon`（预测扩散过程的噪声）或 `v_prediction`（参见 [Imagen Video](https://imagen.research.google/video/paper.pdf) 论文的 2.4 节）
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process)
            or `v_prediction` (see section 2.4 of [Imagen Video](https://imagen.research.google/video/paper.pdf)
            paper).
        timestep_spacing (`str`, defaults to `"leading"`):
            # 时间步的缩放方式。有关更多信息，请参见 [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) 的表 2
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            # 添加到推理步骤的偏移量，一些模型家族需要这个偏移
            An offset added to the inference steps, as required by some model families.
    """
    
    # 定义与 KarrasDiffusionSchedulers 兼容的调度器名称列表
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置默认的调度器顺序
    order = 1

    # 装饰器，用于将初始化函数注册到配置
    @register_to_config
    # 初始化函数
    def __init__(
        # 训练扩散步骤数量，默认为 1000
        self,
        num_train_timesteps: int = 1000,
        # 起始 beta 值，默认为 0.0001
        beta_start: float = 0.0001,
        # 最终 beta 值，默认为 0.02
        beta_end: float = 0.02,
        # beta 调度策略，默认为 "linear"
        beta_schedule: str = "linear",
        # 可选参数，直接传递的 beta 数组
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 是否跳过 Runge-Kutta 步骤，默认为 False
        skip_prk_steps: bool = False,
        # 是否将 alpha 设置为 1，默认为 False
        set_alpha_to_one: bool = False,
        # 预测类型，默认为 "epsilon"
        prediction_type: str = "epsilon",
        # 时间步的缩放方式，默认为 "leading"
        timestep_spacing: str = "leading",
        # 推理步骤的偏移量，默认为 0
        steps_offset: int = 0,
    # 该部分代码为类的方法的一部分
        ):
            # 检查已训练的 beta 参数是否为 None
            if trained_betas is not None:
                # 将训练好的 beta 参数转换为浮点型张量
                self.betas = torch.tensor(trained_betas, dtype=torch.float32)
            # 如果 beta_schedule 为线性调度
            elif beta_schedule == "linear":
                # 生成从 beta_start 到 beta_end 的线性空间，数量为 num_train_timesteps
                self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
            # 如果 beta_schedule 为 scaled_linear
            elif beta_schedule == "scaled_linear":
                # 该调度特定于潜在扩散模型
                # 生成从 beta_start**0.5 到 beta_end**0.5 的线性空间，并平方
                self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
            # 如果 beta_schedule 为 squaredcos_cap_v2
            elif beta_schedule == "squaredcos_cap_v2":
                # 使用 Glide 的余弦调度生成 beta
                self.betas = betas_for_alpha_bar(num_train_timesteps)
            # 如果以上条件都不满足，抛出未实现错误
            else:
                raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")
    
            # 计算 alphas，等于 1 减去 betas
            self.alphas = 1.0 - self.betas
            # 计算 alphas 的累积乘积
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
            # 如果 set_alpha_to_one 为真，则 final_alpha_cumprod 为 1.0，否则取 alphas_cumprod 的第一个值
            self.final_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
    
            # 初始化噪声分布的标准差
            self.init_noise_sigma = 1.0
    
            # 当前只支持 F-PNDM，即龙格-库塔方法
            # 更多算法信息请参考论文：https://arxiv.org/pdf/2202.09778.pdf
            # 主要关注公式 (9), (12), (13) 和算法 2
            self.pndm_order = 4
    
            # 运行时的值初始化
            self.cur_model_output = 0
            self.counter = 0
            self.cur_sample = None
            self.ets = []
    
            # 可设置的值初始化
            self.num_inference_steps = None
            # 创建倒序的时间步数组
            self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
            self.prk_timesteps = None
            self.plms_timesteps = None
            self.timesteps = None
    
        # 定义 step 方法
        def step(
            # 接收模型输出张量
            model_output: torch.Tensor,
            # 当前时间步
            timestep: int,
            # 当前样本张量
            sample: torch.Tensor,
            # 返回字典标志，默认为 True
            return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        预测前一个时间步的样本，通过逆向 SDE 进行。这一函数从学习模型的输出（通常是预测的噪声）中传播扩散过程，
        并根据内部变量 `counter` 调用 [`~PNDMScheduler.step_prk`] 或 [`~PNDMScheduler.step_plms`]。

        参数:
            model_output (`torch.Tensor`):
                来自学习扩散模型的直接输出。
            timestep (`int`):
                当前扩散链中的离散时间步。
            sample (`torch.Tensor`):
                当前通过扩散过程生成的样本实例。
            return_dict (`bool`):
                是否返回 [`~schedulers.scheduling_utils.SchedulerOutput`] 或 `tuple`。

        返回:
            [`~schedulers.scheduling_utils.SchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，返回 [`~schedulers.scheduling_utils.SchedulerOutput`]，否则返回一个
                元组，其中第一个元素是样本张量。

        """
        # 检查当前计数器是否小于 PRK 时间步的长度，且配置中是否跳过 PRK 步骤
        if self.counter < len(self.prk_timesteps) and not self.config.skip_prk_steps:
            # 调用 step_prk 方法，传递模型输出、时间步、样本和返回字典标志
            return self.step_prk(model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict)
        else:
            # 调用 step_plms 方法，传递模型输出、时间步、样本和返回字典标志
            return self.step_plms(model_output=model_output, timestep=timestep, sample=sample, return_dict=return_dict)

    # 定义 step_prk 方法，接收模型输出、时间步、样本和可选的返回字典标志
    def step_prk(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
    # 返回一个调度输出或元组，表示通过逆向SDE预测样本
    ) -> Union[SchedulerOutput, Tuple]:
            """
            通过逆向SDE预测前一个时间步的样本。该函数使用Runge-Kutta方法传播样本。
            进行四次前向传递以逼近微分方程的解。
    
            参数：
                model_output (`torch.Tensor`):
                    来自学习的扩散模型的直接输出。
                timestep (`int`):
                    扩散链中的当前离散时间步。
                sample (`torch.Tensor`):
                    通过扩散过程创建的当前样本实例。
                return_dict (`bool`):
                    是否返回一个[`~schedulers.scheduling_utils.SchedulerOutput`]或元组。
    
            返回：
                [`~schedulers.scheduling_utils.SchedulerOutput`]或`tuple`:
                    如果return_dict为`True`，返回[`~schedulers.scheduling_utils.SchedulerOutput`]，否则返回一个
                    元组，其第一个元素是样本张量。
    
            """
            # 检查推断步骤是否被设置
            if self.num_inference_steps is None:
                raise ValueError(
                    "推断步骤数为'None'，创建调度器后需要运行'set_timesteps'"
                )
    
            # 计算到前一个时间步的差异
            diff_to_prev = 0 if self.counter % 2 else self.config.num_train_timesteps // self.num_inference_steps // 2
            # 确定前一个时间步
            prev_timestep = timestep - diff_to_prev
            # 从预先计算的时间步中获取当前时间步
            timestep = self.prk_timesteps[self.counter // 4 * 4]
    
            # 根据counter的值更新当前模型输出和样本
            if self.counter % 4 == 0:
                self.cur_model_output += 1 / 6 * model_output
                self.ets.append(model_output)
                self.cur_sample = sample
            elif (self.counter - 1) % 4 == 0:
                self.cur_model_output += 1 / 3 * model_output
            elif (self.counter - 2) % 4 == 0:
                self.cur_model_output += 1 / 3 * model_output
            elif (self.counter - 3) % 4 == 0:
                # 更新模型输出并重置当前模型输出
                model_output = self.cur_model_output + 1 / 6 * model_output
                self.cur_model_output = 0
    
            # 确保cur_sample不为`None`
            cur_sample = self.cur_sample if self.cur_sample is not None else sample
    
            # 获取前一个样本
            prev_sample = self._get_prev_sample(cur_sample, timestep, prev_timestep, model_output)
            # 增加计数器
            self.counter += 1
    
            # 根据return_dict返回不同的结果
            if not return_dict:
                return (prev_sample,)
    
            # 返回调度输出
            return SchedulerOutput(prev_sample=prev_sample)
    
        # 定义PLMS步骤方法
        def step_plms(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        预测从上一个时间步生成的样本，通过逆转SDE。该函数使用线性多步法传播样本。
        它多次执行一次前向传递以近似解决方案。

        参数：
            model_output (`torch.Tensor`):
                学习的扩散模型的直接输出。
            timestep (`int`):
                当前扩散链中的离散时间步。
            sample (`torch.Tensor`):
                通过扩散过程生成的当前样本实例。
            return_dict (`bool`):
                是否返回 [`~schedulers.scheduling_utils.SchedulerOutput`] 或元组。

        返回：
            [`~schedulers.scheduling_utils.SchedulerOutput`] 或 `tuple`:
                如果 return_dict 为 `True`，返回 [`~schedulers.scheduling_utils.SchedulerOutput`]，否则返回一个元组，元组的第一个元素是样本张量。

        """
        # 检查推理步骤数是否为 None
        if self.num_inference_steps is None:
            # 抛出错误提示需要设置时间步
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # 检查是否跳过 PRK 步骤，并确保 ETS 列表至少有 3 个元素
        if not self.config.skip_prk_steps and len(self.ets) < 3:
            # 抛出错误提示需要进行至少 12 次迭代
            raise ValueError(
                f"{self.__class__} can only be run AFTER scheduler has been run "
                "in 'prk' mode for at least 12 iterations "
                "See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pipeline_pndm.py "
                "for more information."
            )

        # 计算前一个时间步
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 如果计数器不为 1
        if self.counter != 1:
            # 只保留最近 3 个 ETS 值
            self.ets = self.ets[-3:]
            # 添加当前模型输出到 ETS 列表
            self.ets.append(model_output)
        else:
            # 如果计数器为 1，设置时间步为当前时间步
            prev_timestep = timestep
            timestep = timestep + self.config.num_train_timesteps // self.num_inference_steps

        # 如果 ETS 列表只有 1 个元素且计数器为 0
        if len(self.ets) == 1 and self.counter == 0:
            # 模型输出不变，当前样本为输入样本
            model_output = model_output
            self.cur_sample = sample
        # 如果 ETS 列表只有 1 个元素且计数器为 1
        elif len(self.ets) == 1 and self.counter == 1:
            # 取当前模型输出和最后一个 ETS 的平均值
            model_output = (model_output + self.ets[-1]) / 2
            sample = self.cur_sample
            self.cur_sample = None
        # 如果 ETS 列表有 2 个元素
        elif len(self.ets) == 2:
            # 根据 ETS 计算模型输出
            model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
        # 如果 ETS 列表有 3 个元素
        elif len(self.ets) == 3:
            # 使用更复杂的公式计算模型输出
            model_output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
        # 如果 ETS 列表有 4 个或更多元素
        else:
            # 使用公式计算模型输出，考虑更多的 ETS 值
            model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])

        # 获取前一个样本
        prev_sample = self._get_prev_sample(sample, timestep, prev_timestep, model_output)
        # 增加计数器
        self.counter += 1

        # 如果不返回字典
        if not return_dict:
            # 返回前一个样本的元组
            return (prev_sample,)

        # 返回包含前一个样本的调度器输出
        return SchedulerOutput(prev_sample=prev_sample)
    # 定义缩放模型输入的函数，接收样本和其他参数，返回缩放后的样本
    def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # 文档字符串，说明该函数的作用、参数和返回值
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
    
        Args:
            sample (`torch.Tensor`):
                The input sample.
    
        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        # 直接返回输入样本，不做任何处理
        return sample
    
    # 定义获取前一个样本的函数，基于当前样本、时间步及模型输出
    def _get_prev_sample(self, sample, timestep, prev_timestep, model_output):
        # 参考 PNDM 论文公式 (9)，计算 x_(t−δ)
        # 该函数使用公式 (9) 计算前一个样本
        # 注意需要在方程两边加上 x_t
    
        # 变量注释映射到论文中的符号
        # alpha_prod_t -> α_t
        # alpha_prod_t_prev -> α_(t−δ)
        # beta_prod_t -> (1 - α_t)
        # beta_prod_t_prev -> (1 - α_(t−δ))
        # sample -> x_t
        # model_output -> e_θ(x_t, t)
        # prev_sample -> x_(t−δ)
        # 获取当前时间步的累计 alpha 值
        alpha_prod_t = self.alphas_cumprod[timestep]
        # 获取前一时间步的累计 alpha 值，若无，则使用最终的累计 alpha 值
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        # 计算当前时间步的 beta 值
        beta_prod_t = 1 - alpha_prod_t
        # 计算前一时间步的 beta 值
        beta_prod_t_prev = 1 - alpha_prod_t_prev
    
        # 如果预测类型为 "v_prediction"，根据公式更新模型输出
        if self.config.prediction_type == "v_prediction":
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        # 若预测类型不为 "epsilon"，则抛出异常
        elif self.config.prediction_type != "epsilon":
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon` or `v_prediction`"
            )
    
        # 计算样本系数，对应于公式 (9) 的分子部分
        sample_coeff = (alpha_prod_t_prev / alpha_prod_t) ** (0.5)
    
        # 计算模型输出的分母系数，对应于公式 (9) 的分母部分
        model_output_denom_coeff = alpha_prod_t * beta_prod_t_prev ** (0.5) + (
            alpha_prod_t * beta_prod_t * alpha_prod_t_prev
        ) ** (0.5)
    
        # 根据公式 (9) 计算前一个样本
        prev_sample = (
            sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        )
    
        # 返回计算得到的前一个样本
        return prev_sample
    
        # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的函数
        def add_noise(
            # 定义添加噪声的函数，接收原始样本、噪声和时间步
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 具有与 original_samples 相同的设备和数据类型
        # 将 self.alphas_cumprod 移动到指定设备，以避免后续 add_noise 调用中的 CPU 到 GPU 的冗余数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为与 original_samples 相同的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到与 original_samples 相同的设备
        timesteps = timesteps.to(original_samples.device)

        # 计算 alphas_cumprod 在 timesteps 位置的平方根
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 扁平化为一维张量
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 如果 sqrt_alpha_prod 的维度少于 original_samples，则在最后一个维度增加一个维度
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # 计算 1 - alphas_cumprod 在 timesteps 位置的平方根
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 扁平化为一维张量
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 如果 sqrt_one_minus_alpha_prod 的维度少于 original_samples，则在最后一个维度增加一个维度
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # 根据加权公式生成带噪声的样本
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回带噪声的样本
        return noisy_samples

    # 定义获取对象长度的方法
    def __len__(self):
        # 返回配置中训练时间步的数量
        return self.config.num_train_timesteps
```