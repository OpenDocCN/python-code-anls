# `.\diffusers\pipelines\deprecated\audio_diffusion\pipeline_audio_diffusion.py`

```py
# 版权声明，表明该文件的所有权
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 许可声明，指定该文件遵循的许可证类型
# Licensed under the Apache License, Version 2.0 (the "License");
# 规定用户不得在不遵循许可证的情况下使用此文件
# you may not use this file except in compliance with the License.
# 提供许可证获取链接
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 免责声明，声明软件在“按原样”基础上分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何明示或暗示的担保或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 指出许可证的具体条款和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入数学库中的反余弦函数和正弦函数
from math import acos, sin
# 导入类型提示相关的类
from typing import List, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入图像处理库
from PIL import Image

# 从模型模块导入自编码器和 UNet 模型
from ....models import AutoencoderKL, UNet2DConditionModel
# 从调度器模块导入调度器类
from ....schedulers import DDIMScheduler, DDPMScheduler
# 从工具模块导入随机张量生成函数
from ....utils.torch_utils import randn_tensor
# 从管道工具模块导入音频输出和基本输出类
from ...pipeline_utils import AudioPipelineOutput, BaseOutput, DiffusionPipeline, ImagePipelineOutput
# 从音频处理模块导入 Mel 类
from .mel import Mel

# 定义音频扩散管道类，继承自扩散管道基类
class AudioDiffusionPipeline(DiffusionPipeline):
    """
    音频扩散管道。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档，以获取所有管道实现的通用方法（下载、保存、在特定设备上运行等）。

    参数：
        vqae ([`AutoencoderKL`]):
            用于编码和解码图像到潜在表示的变分自编码器（VAE）模型。
        unet ([`UNet2DConditionModel`]):
            用于对编码图像潜在值去噪的 `UNet2DConditionModel`。
        mel ([`Mel`]):
            将音频转换为声谱图。
        scheduler ([`DDIMScheduler`] 或 [`DDPMScheduler`]):
            与 `unet` 一起使用的调度器，用于对编码图像潜在值去噪。可以是 [`DDIMScheduler`] 或 [`DDPMScheduler`] 中的任意一种。
    """

    # 定义可选组件的名称列表
    _optional_components = ["vqvae"]

    # 初始化方法，定义所需的参数
    def __init__(
        self,
        vqvae: AutoencoderKL,  # 变分自编码器模型
        unet: UNet2DConditionModel,  # UNet 模型
        mel: Mel,  # Mel 转换器
        scheduler: Union[DDIMScheduler, DDPMScheduler],  # 调度器
    ):
        super().__init__()  # 调用父类的初始化方法
        # 注册模型和组件
        self.register_modules(unet=unet, scheduler=scheduler, mel=mel, vqvae=vqvae)

    # 获取默认步骤数的方法
    def get_default_steps(self) -> int:
        """返回推荐的推理默认步骤数。

        返回:
            `int`:
                步骤数。
        """
        # 根据调度器类型返回对应的步骤数
        return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

    # 装饰器，指示以下方法不需要计算梯度
    @torch.no_grad()
    # 定义可调用对象，支持多种参数配置
        def __call__(
            self,  # 该方法可以被直接调用
            batch_size: int = 1,  # 批次大小，默认为 1
            audio_file: str = None,  # 音频文件的路径，默认为 None
            raw_audio: np.ndarray = None,  # 原始音频数据，默认为 None
            slice: int = 0,  # 切片起始位置，默认为 0
            start_step: int = 0,  # 起始步数，默认为 0
            steps: int = None,  # 总步数，默认为 None
            generator: torch.Generator = None,  # 随机数生成器，默认为 None
            mask_start_secs: float = 0,  # 掩码开始时间（秒），默认为 0
            mask_end_secs: float = 0,  # 掩码结束时间（秒），默认为 0
            step_generator: torch.Generator = None,  # 步数生成器，默认为 None
            eta: float = 0,  # 噪声控制参数，默认为 0
            noise: torch.Tensor = None,  # 噪声张量，默认为 None
            encoding: torch.Tensor = None,  # 编码张量，默认为 None
            return_dict=True,  # 是否返回字典格式的结果，默认为 True
        ) -> Union[  # 返回值的类型说明
            Union[AudioPipelineOutput, ImagePipelineOutput],  # 可返回音频或图像管道输出
            Tuple[List[Image.Image], Tuple[int, List[np.ndarray]]],  # 也可返回图像列表和元组
        @torch.no_grad()  # 关闭梯度计算以节省内存
        def encode(self, images: List[Image.Image], steps: int = 50) -> np.ndarray:  # 编码方法
            """
            反向去噪过程以恢复生成图像的噪声图像。
    
            参数:
                images (`List[PIL Image]`):  # 输入图像列表
                    要编码的图像列表。
                steps (`int`):  # 编码步数
                    执行的编码步数（默认为 `50`）。
    
            返回:
                `np.ndarray`:  # 返回的噪声张量
                    形状为 `(batch_size, 1, height, width)` 的噪声张量。
            """
    
            # 仅适用于 DDIM，因为此方法是确定性的
            assert isinstance(self.scheduler, DDIMScheduler)  # 确保调度器是 DDIM 类型
            self.scheduler.set_timesteps(steps)  # 设置调度器的时间步
            sample = np.array(  # 将输入图像转换为 NumPy 数组
                [np.frombuffer(image.tobytes(), dtype="uint8").reshape((1, image.height, image.width)) for image in images]  # 读取图像数据并重塑为适当形状
            )
            sample = (sample / 255) * 2 - 1  # 归一化样本到 [-1, 1] 范围
            sample = torch.Tensor(sample).to(self.device)  # 转换为 PyTorch 张量并移动到设备上
    
            for t in self.progress_bar(torch.flip(self.scheduler.timesteps, (0,))):  # 迭代时间步的反向顺序
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps  # 计算前一个时间步
                alpha_prod_t = self.scheduler.alphas_cumprod[t]  # 获取当前时间步的累积 alpha 值
                alpha_prod_t_prev = (  # 获取前一个时间步的累积 alpha 值
                    self.scheduler.alphas_cumprod[prev_timestep]  # 如果存在，则使用前一个时间步
                    if prev_timestep >= 0
                    else self.scheduler.final_alpha_cumprod  # 否则使用最终的累积 alpha 值
                )
                beta_prod_t = 1 - alpha_prod_t  # 计算 beta 值
                model_output = self.unet(sample, t)["sample"]  # 通过 UNet 模型获取输出样本
                pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output  # 预测样本方向
                sample = (sample - pred_sample_direction) * alpha_prod_t_prev ** (-0.5)  # 更新样本
                sample = sample * alpha_prod_t ** (0.5) + beta_prod_t ** (0.5) * model_output  # 应用当前时间步的更新
    
            return sample  # 返回处理后的样本
    
        @staticmethod  # 标记为静态方法
    # 定义球面线性插值函数，接受两个张量和一个插值因子
    def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) -> torch.Tensor:
        """Spherical Linear intERPolation.
    
        Args:
            x0 (`torch.Tensor`):
                第一个用于插值的张量。
            x1 (`torch.Tensor`):
                第二个用于插值的张量。
            alpha (`float`):
                插值因子，范围在 0 到 1 之间
    
        Returns:
            `torch.Tensor`:
                插值后的张量。
        """
    
        # 计算两个张量之间的夹角 theta，使用余弦定理
        theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) / torch.norm(x1))
        # 根据球面线性插值公式计算并返回插值结果
        return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)
```