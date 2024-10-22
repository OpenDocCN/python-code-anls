# `.\diffusers\schedulers\scheduling_ddpm_parallel.py`

```py
# 版权声明，说明文件的版权所有者及许可信息
# Copyright 2024 ParaDiGMS authors and The HuggingFace Team. All rights reserved.
#
# 根据 Apache License, Version 2.0 授权使用该文件
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# 你可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，否则以 "AS IS" 基础分发该文件，没有任何形式的担保
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 查看许可证以了解特定的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 声明该文件受 https://github.com/ermongroup/ddim 的影响
# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim

# 导入数学库
import math
# 导入数据类装饰器
from dataclasses import dataclass
# 导入类型注解
from typing import List, Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch

# 从配置工具导入混合类和注册函数
from ..configuration_utils import ConfigMixin, register_to_config
# 从工具导入基础输出类
from ..utils import BaseOutput
# 从 PyTorch 工具导入随机张量函数
from ..utils.torch_utils import randn_tensor
# 从调度工具导入调度器和混合类
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin


@dataclass
# 定义用于调度器输出的类，继承自基础输出类
class DDPMParallelSchedulerOutput(BaseOutput):
    """
    调度器的 `step` 函数输出的输出类。

    参数：
        prev_sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)` 的图像):
            先前时间步的计算样本 `(x_{t-1})`。 `prev_sample` 应用作下一次模型输入
            在去噪循环中。
        pred_original_sample (`torch.Tensor` 形状为 `(batch_size, num_channels, height, width)` 的图像):
            基于当前时间步的模型输出的预测去噪样本 `(x_{0})`。
            `pred_original_sample` 可用于预览进度或进行指导。
    """

    # 先前样本的张量
    prev_sample: torch.Tensor
    # 可选的预测去噪样本张量
    pred_original_sample: Optional[torch.Tensor] = None


# 从贝塔计算函数复制的定义
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    创建一个贝塔调度，离散化给定的 alpha_t_bar 函数，该函数定义了时间 t=[0,1] 的 (1-beta) 的累积乘积。

    包含一个 alpha_bar 函数，该函数接收参数 t 并转换为 (1-beta) 在扩散过程中的累积乘积。

    参数：
        num_diffusion_timesteps (`int`): 要生成的贝塔数量。
        max_beta (`float`): 要使用的最大贝塔值；使用小于 1 的值以防止奇异性。
        alpha_transform_type (`str`, *可选*, 默认为 `cosine`): alpha_bar 的噪声调度类型。
                     选择 `cosine` 或 `exp`

    返回：
        betas (`np.ndarray`): 调度器用于逐步模型输出的贝塔值
    """
    # 检查 alpha_transform_type 是否为 "cosine"
    if alpha_transform_type == "cosine":

        # 定义 alpha_bar_fn 函数，计算 cos 变换
        def alpha_bar_fn(t):
            # 返回 cos 函数值的平方，用于 alpha 变换
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    # 检查 alpha_transform_type 是否为 "exp"
    elif alpha_transform_type == "exp":

        # 定义 alpha_bar_fn 函数，计算指数衰减
        def alpha_bar_fn(t):
            # 返回指数衰减值，用于 alpha 变换
            return math.exp(t * -12.0)

    # 如果 alpha_transform_type 不符合预期，抛出异常
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    # 初始化空列表，用于存储 beta 值
    betas = []
    # 遍历每一个扩散时间步
    for i in range(num_diffusion_timesteps):
        # 计算当前时间步的比例 t1
        t1 = i / num_diffusion_timesteps
        # 计算下一个时间步的比例 t2
        t2 = (i + 1) / num_diffusion_timesteps
        # 计算 beta 值，并加入到 betas 列表中，取最小值以限制在 max_beta 之内
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    # 将 betas 列表转换为 PyTorch 张量，并指定数据类型为 float32
    return torch.tensor(betas, dtype=torch.float32)
# 从 diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr 复制而来
def rescale_zero_terminal_snr(betas):
    """
    根据 https://arxiv.org/pdf/2305.08891.pdf (算法 1) 调整 betas 使其具有零终端 SNR

    参数:
        betas (`torch.Tensor`):
            初始化调度器时使用的 betas。

    返回:
        `torch.Tensor`: 调整后的具有零终端 SNR 的 betas
    """
    # 将 betas 转换为 alphas_bar_sqrt
    alphas = 1.0 - betas  # 计算 alphas，表示每个时间步的保留比率
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # 计算 alphas 的累积乘积
    alphas_bar_sqrt = alphas_cumprod.sqrt()  # 计算累积乘积的平方根

    # 存储旧值
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()  # 记录初始值
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()  # 记录最后值

    # 移动，使最后一个时间步为零
    alphas_bar_sqrt -= alphas_bar_sqrt_T  # 将最后时间步的值减去

    # 缩放，使第一个时间步恢复到旧值
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)  # 调整比例

    # 将 alphas_bar_sqrt 转换回 betas
    alphas_bar = alphas_bar_sqrt**2  # 还原平方根
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # 还原累积乘积
    alphas = torch.cat([alphas_bar[0:1], alphas])  # 将第一个元素加入 alphas
    betas = 1 - alphas  # 计算新的 betas

    return betas  # 返回调整后的 betas


class DDPMParallelScheduler(SchedulerMixin, ConfigMixin):
    """
    去噪扩散概率模型 (DDPM) 探索去噪评分匹配与 Langevin 动力学采样之间的联系。

    [`~ConfigMixin`] 负责存储调度器 `__init__` 函数中传入的所有配置属性，例如 `num_train_timesteps`。
    这些属性可以通过 `scheduler.config.num_train_timesteps` 访问。
    [`SchedulerMixin`] 提供通用的加载和保存功能，通过 [`SchedulerMixin.save_pretrained`] 和
    [`~SchedulerMixin.from_pretrained`] 函数。

    更多详细信息，请参阅原始论文: https://arxiv.org/abs/2006.11239
    # 参数说明
    Args:
        # 训练模型所用的扩散步骤数量
        num_train_timesteps (`int`): number of diffusion steps used to train the model.
        # 推理时的起始 `beta` 值
        beta_start (`float`): the starting `beta` value of inference.
        # 推理时的最终 `beta` 值
        beta_end (`float`): the final `beta` value.
        # `beta` 调度策略
        beta_schedule (`str`):
            # 从 `beta` 范围映射到一系列 `beta` 值，用于模型的步骤选择
            the beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, `squaredcos_cap_v2` or `sigmoid`.
        # 可选参数，直接传递 `beta` 数组到构造函数，绕过 `beta_start` 和 `beta_end`
        trained_betas (`np.ndarray`, optional):
            option to pass an array of betas directly to the constructor to bypass `beta_start`, `beta_end` etc.
        # 噪声添加时的方差裁剪选项
        variance_type (`str`):
            # 噪声添加时方差的裁剪选项，选择包括 `fixed_small`、`fixed_small_log`、`fixed_large`、`fixed_large_log`、`learned` 或 `learned_range`
            options to clip the variance used when adding noise to the denoised sample. Choose from `fixed_small`,
            `fixed_small_log`, `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        # 是否裁剪预测样本以确保数值稳定性
        clip_sample (`bool`, default `True`):
            option to clip predicted sample for numerical stability.
        # 样本裁剪的最大幅度，仅在 `clip_sample=True` 时有效
        clip_sample_range (`float`, default `1.0`):
            the maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        # 调度函数的预测类型
        prediction_type (`str`, default `epsilon`, optional):
            # 调度函数的预测类型，选择包括 `epsilon`（预测扩散过程的噪声）、`sample`（直接预测噪声样本）或 `v_prediction`
            prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion
            process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4
            https://imagen.research.google/video/paper.pdf)
        # 是否使用“动态阈值”方法
        thresholding (`bool`, default `False`):
            # 是否使用“动态阈值”方法，该方法由 Imagen 引入
            whether to use the "dynamic thresholding" method (introduced by Imagen, https://arxiv.org/abs/2205.11487).
            Note that the thresholding method is unsuitable for latent-space diffusion models (such as
            stable-diffusion).
        # 动态阈值方法的比例，仅在 `thresholding=True` 时有效
        dynamic_thresholding_ratio (`float`, default `0.995`):
            # 动态阈值方法的比例，默认值为 `0.995`
            the ratio for the dynamic thresholding method. Default is `0.995`, the same as Imagen
            (https://arxiv.org/abs/2205.11487). Valid only when `thresholding=True`.
        # 动态阈值的阈值值，仅在 `thresholding=True` 时有效
        sample_max_value (`float`, default `1.0`):
            # 动态阈值的阈值值，默认值为 `1.0`
            the threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        # 时间步长的缩放方式
        timestep_spacing (`str`, default `"leading"`):
            # 时间步长的缩放方式，参考文献提供的表格
            The way the timesteps should be scaled. Refer to Table 2. of [Common Diffusion Noise Schedules and Sample
            Steps are Flawed](https://arxiv.org/abs/2305.08891) for more information.
        # 推理步骤的偏移量
        steps_offset (`int`, default `0`):
            # 添加到推理步骤的偏移量，一些模型族可能需要
            An offset added to the inference steps, as required by some model families.
        # 是否将 `beta` 重新缩放到零终端 SNR
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            # 是否将 `beta` 重新缩放，以实现零终端 SNR
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """
    # 获取 KarrasDiffusionSchedulers 中所有调度器的名称，形成兼容列表
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    # 设置调度器的顺序
    order = 1
    # 标记是否为 ODE 调度器
    _is_ode_scheduler = False

    @register_to_config
    # 定义初始化函数，配置调度器参数
    def __init__(
        # 训练时间步的数量，默认值为 1000
        num_train_timesteps: int = 1000,
        # 初始 beta 值，默认值为 0.0001
        beta_start: float = 0.0001,
        # 结束 beta 值，默认值为 0.02
        beta_end: float = 0.02,
        # beta 调度方式，默认值为 "linear"
        beta_schedule: str = "linear",
        # 训练好的 beta 值，可以是 ndarray 或浮点数列表
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        # 方差类型，默认值为 "fixed_small"
        variance_type: str = "fixed_small",
        # 是否裁剪样本，默认值为 True
        clip_sample: bool = True,
        # 预测类型，默认值为 "epsilon"
        prediction_type: str = "epsilon",
        # 是否进行阈值处理，默认值为 False
        thresholding: bool = False,
        # 动态阈值处理的比率，默认值为 0.995
        dynamic_thresholding_ratio: float = 0.995,
        # 裁剪样本的范围，默认值为 1.0
        clip_sample_range: float = 1.0,
        # 时间步间隔类型，默认值为 "leading"
        timestep_spacing: str = "leading",
        # 步骤偏移量，默认值为 0
        steps_offset: int = 0,
        # 是否对零 SNR 进行重新缩放，默认值为 False
        rescale_betas_zero_snr: bool = False,
    ):
        # 如果提供了训练好的 beta 值，则将其转换为 tensor
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        # 如果使用线性调度，则生成线性变化的 beta 值
        elif beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        # 如果使用缩放线性调度，特定于潜在扩散模型
        elif beta_schedule == "scaled_linear":
            # 生成缩放后的 beta 值
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32) ** 2
        # 如果使用平方余弦调度，生成对应的 beta 值
        elif beta_schedule == "squaredcos_cap_v2":
            # 使用 Glide 余弦调度
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        # 如果使用 sigmoid 调度，生成对应的 beta 值
        elif beta_schedule == "sigmoid":
            # 使用 GeoDiff 的 sigmoid 调度
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        # 如果调度方式未实现，抛出未实现错误
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        # 如果需要，重新缩放 beta 值以适应零 SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        # 计算 alpha 值为 1 减去 beta 值
        self.alphas = 1.0 - self.betas
        # 计算累积 alpha 值的乘积
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # 创建一个常数 tensor，值为 1.0
        self.one = torch.tensor(1.0)

        # 初始化噪声分布的标准差，默认值为 1.0
        self.init_noise_sigma = 1.0

        # 可设置的值，初始化为 False
        self.custom_timesteps = False
        # 进行推理步骤的数量，初始化为 None
        self.num_inference_steps = None
        # 创建时间步的 tensor，范围从 0 到 num_train_timesteps，反向排序
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

        # 设置方差类型
        self.variance_type = variance_type

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.scale_model_input 复制的函数
    # 定义一个方法，用于根据当前时间步调整模型输入的大小
    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """
        确保与需要根据当前时间步缩放去噪模型输入的调度器之间的互换性。

        参数:
            sample (`torch.Tensor`):
                输入样本。
            timestep (`int`, *可选*):
                扩散链中的当前时间步。

        返回:
            `torch.Tensor`:
                一个缩放后的输入样本。
        """
        # 直接返回输入样本，未进行任何缩放处理
        return sample

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.set_timesteps 复制
    def set_timesteps(
        # 定义一个可选参数，表示推理步骤的数量
        num_inference_steps: Optional[int] = None,
        # 定义一个可选参数，表示设备类型
        device: Union[str, torch.device] = None,
        # 定义一个可选参数，表示时间步的列表
        timesteps: Optional[List[int]] = None,
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._get_variance 复制
    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        # 获取当前时间步的前一个时间步
        prev_t = self.previous_timestep(t)

        # 获取当前时间步的累积 alpha 值
        alpha_prod_t = self.alphas_cumprod[t]
        # 获取前一个时间步的累积 alpha 值，如果前一个时间步小于 0，则使用默认值 one
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        # 计算当前时间步的 beta 值
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # 对于 t > 0，计算预测的方差 βt（参见 https://arxiv.org/pdf/2006.11239.pdf 中的公式 (6) 和 (7)）
        # 从中进行采样以获得前一个样本
        # x_{t-1} ~ N(pred_prev_sample, variance) == 将方差添加到预测样本
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # 始终取方差的对数，因此需要限制它以确保不为 0
        variance = torch.clamp(variance, min=1e-20)

        # 如果方差类型未指定，则使用配置中定义的方差类型
        if variance_type is None:
            variance_type = self.config.variance_type

        # 针对不同的方差类型进行处理
        # 固定小方差的情况
        if variance_type == "fixed_small":
            variance = variance
        # 针对 rl-diffuser 的情况 https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            # 对方差取对数后取指数以计算最终方差
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            # 将方差设置为当前 beta 值
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # 对当前 beta 值取对数
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            # 返回预测的方差
            return predicted_variance
        elif variance_type == "learned_range":
            # 计算最小和最大方差的对数
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            # 计算 frac 以进行线性插值
            frac = (predicted_variance + 1) / 2
            # 通过线性插值计算方差
            variance = frac * max_log + (1 - frac) * min_log

        # 返回计算得到的方差
        return variance

    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample 复制
    # 定义私有方法 _threshold_sample，接受一个张量类型的样本并返回处理后的张量
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "动态阈值：在每次采样步骤中，我们将 s 设置为 xt0（在时间步 t 对 x_0 的预测）中的某个百分位绝对像素值，
        如果 s > 1，则我们将 xt0 的值阈值化到范围 [-s, s]，然后除以 s。动态阈值推动饱和像素（接近 -1 和 1 的像素）向内移动，
        从而在每一步主动防止像素饱和。我们发现动态阈值显著提高了照片真实感以及图像与文本的对齐，尤其是在使用非常大的引导权重时。"

        https://arxiv.org/abs/2205.11487
        """
        # 获取样本的数值类型
        dtype = sample.dtype
        # 获取批次大小、通道数和剩余维度
        batch_size, channels, *remaining_dims = sample.shape

        # 如果样本类型不是 float32 或 float64，则转换为 float 类型
        if dtype not in (torch.float32, torch.float64):
            sample = sample.float()  # 为量化计算进行上升类型转换，且 cpu 半精度不支持 clamping

        # 将样本展平，以便在每张图像上进行百分位计算
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        # 计算样本的绝对值
        abs_sample = sample.abs()  # "某个百分位绝对像素值"

        # 计算绝对样本的指定百分位数
        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        # 将 s 的值限制在 [1, sample_max_value] 范围内
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # 限制最小值为 1，相当于标准裁剪到 [-1, 1]
        # 在第一维度上添加一个维度以便广播
        s = s.unsqueeze(1)  # (batch_size, 1)，因为 clamp 会在第 0 维广播
        # 将样本值限制在 [-s, s] 范围内，并归一化
        sample = torch.clamp(sample, -s, s) / s  # "将 xt0 阈值化到范围 [-s, s]，然后除以 s"

        # 将样本恢复到原始形状
        sample = sample.reshape(batch_size, channels, *remaining_dims)
        # 将样本转换回原来的数值类型
        sample = sample.to(dtype)

        # 返回处理后的样本
        return sample

    # 定义步进方法 step，接受模型输出、时间步、样本等参数
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    # 定义无噪声的批处理步骤方法 batch_step_no_noise
    def batch_step_no_noise(
        self,
        model_output: torch.Tensor,
        timesteps: List[int],
        sample: torch.Tensor,
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.add_noise 复制的方法 add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    # 返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 的设备和数据类型与 original_samples 相同
        # 将 self.alphas_cumprod 移动到目标设备，以避免后续 add_noise 调用时重复的 CPU 到 GPU 数据移动
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        # 将 alphas_cumprod 转换为 original_samples 的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        # 将 timesteps 移动到 original_samples 的设备
        timesteps = timesteps.to(original_samples.device)
    
        # 计算平方根的 alpha 累积乘积
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 确保 sqrt_alpha_prod 的维度与 original_samples 相同
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
        # 计算平方根的 1 - alpha 累积乘积
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 确保 sqrt_one_minus_alpha_prod 的维度与 original_samples 相同
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
        # 通过加噪声生成带噪声的样本
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        # 返回带噪声的样本
        return noisy_samples
    
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.get_velocity 复制
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        # 确保 alphas_cumprod 和 timestep 的设备和数据类型与 sample 相同
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        # 将 alphas_cumprod 转换为 sample 的数据类型
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        # 将 timesteps 移动到 sample 的设备
        timesteps = timesteps.to(sample.device)
    
        # 计算平方根的 alpha 累积乘积
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        # 将 sqrt_alpha_prod 展平
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        # 确保 sqrt_alpha_prod 的维度与 sample 相同
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
        # 计算平方根的 1 - alpha 累积乘积
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        # 将 sqrt_one_minus_alpha_prod 展平
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        # 确保 sqrt_one_minus_alpha_prod 的维度与 sample 相同
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
    
        # 计算速度，即带噪声的样本与原始样本的差异
        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        # 返回计算得到的速度
        return velocity
    
    # 定义获取对象长度的方法
    def __len__(self):
        # 返回训练时间步的数量
        return self.config.num_train_timesteps
    
    # 从 diffusers.schedulers.scheduling_ddpm.DDPMScheduler.previous_timestep 复制
    # 获取给定时间步的前一个时间步
        def previous_timestep(self, timestep):
            # 检查是否使用自定义时间步
            if self.custom_timesteps:
                # 查找当前时间步在时间步数组中的索引
                index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
                # 如果当前时间步是最后一个时间步
                if index == self.timesteps.shape[0] - 1:
                    # 将前一个时间步设置为 -1
                    prev_t = torch.tensor(-1)
                else:
                    # 否则，取当前时间步之后的时间步作为前一个时间步
                    prev_t = self.timesteps[index + 1]
            else:
                # 根据推理步骤数量确定前一个时间步的计算方式
                num_inference_steps = (
                    # 如果已定义推理步骤数量，则使用该值，否则使用训练时间步数量
                    self.num_inference_steps if self.num_inference_steps else self.config.num_train_timesteps
                )
                # 计算前一个时间步
                prev_t = timestep - self.config.num_train_timesteps // num_inference_steps
    
            # 返回前一个时间步
            return prev_t
```