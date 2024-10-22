# `.\diffusers\schedulers\scheduling_ddim_inverse.py`

```py
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 许可声明，指明版权及使用条件
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 表示软件是按现状提供，不提供任何形式的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 参阅许可协议以了解具体权限和限制

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion
# 声明代码受相关项目的影响，提供了代码来源信息

import math
# 导入数学库，用于数学计算
from dataclasses import dataclass
# 从dataclasses模块导入dataclass装饰器，用于简化类的定义
from typing import List, Optional, Tuple, Union
# 从typing模块导入类型注解，用于类型提示

import numpy as np
# 导入NumPy库，用于数值计算和数组操作
import torch
# 导入PyTorch库，用于深度学习

from diffusers.configuration_utils import ConfigMixin, register_to_config
# 从diffusers.configuration_utils导入ConfigMixin和register_to_config，用于配置管理
from diffusers.schedulers.scheduling_utils import SchedulerMixin
# 从diffusers.schedulers.scheduling_utils导入SchedulerMixin，用于调度器的混合功能
from diffusers.utils import BaseOutput, deprecate
# 从diffusers.utils导入BaseOutput和deprecate，用于输出格式和弃用警告

@dataclass
# 使用dataclass装饰器定义数据类
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM
class DDIMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.
    # 定义调度器的`step`函数输出类

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
            # 前一时间步计算的样本，作为下一步去噪循环的输入
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
            # 当前时间步模型输出的预测去噪样本，可以用于进度预览或指导
    """

    prev_sample: torch.Tensor
    # 定义前一时间步样本，类型为torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
    # 定义预测去噪样本，类型为可选的torch.Tensor，默认为None

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].
    # 创建一个beta调度，离散化给定的alpha_t_bar函数，定义(1-beta)随时间的累积乘积

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    # 包含一个函数alpha_bar，接受参数t并将其转换为扩散过程的(1-beta)累积乘积

    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        # 生成的beta数量
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        # 使用的最大beta值，避免使用1以上的值以防止奇点
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`
        # 噪声调度类型，选择'cosine'或'exp'

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    # 返回调度器用于模型输出的betas
    """
    # 检查 alpha_transform_type 是否为 "cosine"
    if alpha_transform_type == "cosine":

        # 定义一个函数，计算基于余弦函数的 alpha 值
        def alpha_bar_fn(t):
            # 计算并返回余弦平方值，调整参数以确保值在特定范围
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 检查 alpha_transform_type 是否为 "exp"
    elif alpha_transform_type == "exp":

        # 定义一个函数，计算基于指数函数的 alpha 值
        def alpha_bar_fn(t):
            # 计算并返回负指数值，参数设为 -12.0
            return math.exp(t * -12.0)

    # 如果 alpha_transform_type 不符合预期，抛出异常
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化一个空列表以存储 beta 值
    betas = []
    # 遍历每个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步的比例 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步的比例 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值并添加到列表，确保不超过 max_beta
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 beta 列表转换为浮点型的张量并返回
    return torch.tensor(betas, dtype=torch.float32)
# 从 diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr 复制而来
def rescale_zero_terminal_snr(betas):
    """
    根据 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 将 betas 重新缩放为具有零终端 SNR

    参数：
        betas (`torch.Tensor`):
            用于初始化调度器的 betas。

    返回：
        `torch.Tensor`: 具有零终端 SNR 的重新缩放的 betas
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas，表示每个 beta 对应的 alpha
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 对累积乘积取平方根

    # 存储旧值
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 记录初始值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 记录终止值

    # 移动，使最后一个时间步为零
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 从 alphas_bar_sqrt 中减去终止值

    # 缩放，使第一个时间步恢复到旧值
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 根据初始值和终止值进行缩放

    # 将 alphas_bar_sqrt 转换回 betas
    alphas_bar = alphas_bar_sqrt**2  # 将平方根还原为平方
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 根据累积乘积的逆操作恢复 alphas
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个 alpha 加入到 alphas 中
    betas = 1 - alphas  # 计算 betas，取 1 减去 alphas

    return betas  # 返回重新缩放的 betas


class DDIMInverseScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDIMInverseScheduler` 是 [`DDIMScheduler`] 的反向调度器。

    该模型继承自 [`SchedulerMixin`] 和 [`ConfigMixin`]。请查看父类文档，以了解库为所有调度器实现的通用方法，例如加载和保存。
    # 定义参数文档字符串，描述每个参数的作用和默认值
    Args:
        # 训练模型的扩散步骤数，默认值为1000
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        # 推理的起始 beta 值，默认值为 0.0001
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        # 最终的 beta 值，默认值为 0.02
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        # beta 调度类型，默认为 "linear"，用于映射 beta 范围到模型的步进序列
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        # 直接传入 beta 数组以绕过 beta_start 和 beta_end 的设置，属于可选参数
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        # 是否对预测样本进行剪裁以提高数值稳定性，默认值为 True
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        # 样本剪裁的最大幅度，仅在 clip_sample=True 时有效，默认值为 1.0
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        # 每个扩散步骤使用当前和上一个步骤的 alphas 乘积值，最后一步的前一个 alpha 固定为 0 的选项，默认值为 True
        set_alpha_to_one (`bool`, defaults to `True`):
            Each diffusion step uses the alphas product value at that step and at the previous one. For the final step
            there is no previous alpha. When this option is `True` the previous alpha product is fixed to 0, otherwise
            it uses the alpha value at step `num_train_timesteps - 1`.
        # 推理步骤的偏移量，某些模型族可能需要，默认值为 0
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        # 调度函数的预测类型，可选项包括 `epsilon`（预测扩散过程的噪声）、`sample`（直接预测噪声样本）或 `v_prediction`
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        # 时间步的缩放方式，默认值为 "leading"
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        # 是否将 beta 重新缩放为零终端 SNR，允许模型生成非常明亮和非常暗的样本，默认值为 False
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    # 参数文档字符串结束

    # 设置参数的顺序
    order = 1
    # 指定在配置中忽略的参数
    ignore_for_config = ["kwargs"]
    # 标记已弃用的参数
    _deprecated_kwargs = ["set_alpha_to_zero"]

    # 注册到配置
    @register_to_config
    # 初始化函数，设置了多个参数和默认值，包括训练步数、beta的起始和结束值、beta的调度类型、
    # 训练后的betas数组、是否剪裁样本、是否将alpha设置为1、步骤偏移量、预测类型、剪裁样本范围、
    # 时间步间距、是否对零信噪比重新缩放等
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        clip_sample_range: float = 1.0,
        timestep_spacing: str = "leading",
        rescale_betas_zero_snr: bool = False,
        **kwargs,
    ):
        # 如果kwargs中存在"set_alpha_to_zero"参数，则给出警告信息并使用其值替代"set_alpha_to_one"参数
        if kwargs.get("set_alpha_to_zero", None) is not None:
            deprecation_message = (
                "The `set_alpha_to_zero` argument is deprecated. Please use `set_alpha_to_one` instead."
            )
            deprecate("set_alpha_to_zero", "1.0.0", deprecation_message, standard_warn=False)
            set_alpha_to_one = kwargs["set_alpha_to_zero"]
        
        # 如果给定了训练后的betas数组，则使用它来初始化self.betas
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果beta_schedule为"linear"，则使用线性插值生成self.betas数组
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果beta_schedule为"scaled_linear"，则使用特定的缩放线性插值生成self.betas数组
        elif beta_schedule == "scaled_linear":
            # 此调度方式非常特定于潜在扩散模型。
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 如果beta_schedule为"squaredcos_cap_v2"，则使用特定的函数生成self.betas数组
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            # 抛出未实现的调度类型异常
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 如果设置了rescale_betas_zero_snr标志为True，则对self.betas数组进行零信噪比重新缩放
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        # 根据self.betas计算self.alphas，并计算累积乘积
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # 在反向ddim中的每个步骤中，我们查看下一个alphas_cumprod
        # 对于初始步骤，没有当前的alphas_cumprod，索引越界
        # `set_alpha_to_one`决定是否将此参数简单设置为1
        # 在这种情况下，self.step()仅输出预测的噪声
        # 或者是否使用训练扩散模型中使用的初始alpha。
        self.initial_alpha_cumprod = torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]

        # 初始噪声分布的标准差
        self.init_noise_sigma = 1.0

        # 可设置的值
        self.num_inference_steps = None
        # 使用np.arange创建时间步长的张量
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps).copy().astype(np.int64))
    # 定义一个方法，缩放模型输入以确保与调度器的互换性
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        确保根据当前时间步缩放去噪模型输入，以便与需要的调度器互换。
    
        参数:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。
    
        返回:
            `torch.Tensor`:
                缩放后的输入样本。
        """
        # 返回输入样本，当前未对其进行缩放
        return sample
    
    # 定义一个方法，设置用于扩散链的离散时间步（在推理前运行）
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        设置用于扩散链的离散时间步（在推理前运行）。
    
        参数:
            num_inference_steps (`int`):
                生成样本时使用的扩散步骤数量。
        """
    
        # 检查推理步骤数量是否超过训练时的最大时间步数
        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} 不能大于 `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} 因为使用该调度器训练的 unet 模型最多只能处理"
                f" {self.config.num_train_timesteps} 个时间步。"
            )
    
        # 将推理步骤数量赋值给实例变量
        self.num_inference_steps = num_inference_steps
    
        # 根据配置的时间步间隔设置时间步
        if self.config.timestep_spacing == "leading":
            # 计算每步的比例
            step_ratio = self.config.num_train_timesteps // self.num_inference_steps
            # 通过乘以比例创建整数时间步
            # 转换为整数以避免当 num_inference_steps 是 3 的幂时出现问题
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
            # 加上步骤偏移量
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            # 计算每步的比例
            step_ratio = self.config.num_train_timesteps / self.num_inference_steps
            # 通过乘以比例创建整数时间步
            # 转换为整数以避免当 num_inference_steps 是 3 的幂时出现问题
            timesteps = np.round(np.arange(self.config.num_train_timesteps, 0, -step_ratio)[::-1]).astype(np.int64)
            # 减去 1
            timesteps -= 1
        else:
            # 如果时间步间隔不支持，抛出错误
            raise ValueError(
                f"{self.config.timestep_spacing} 不被支持。请确保选择 'leading' 或 'trailing' 之一。"
            )
    
        # 将时间步转换为张量并移动到指定设备
        self.timesteps = torch.from_numpy(timesteps).to(device)
    
    # 定义一个方法，执行一步操作
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        return_dict: bool = True,
    ):
        # 定义一个方法，返回训练时的时间步数量
        def __len__(self):
            return self.config.num_train_timesteps
```