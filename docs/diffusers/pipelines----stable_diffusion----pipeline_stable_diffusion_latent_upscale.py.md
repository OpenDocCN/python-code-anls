# `.\diffusers\pipelines\stable_diffusion\pipeline_stable_diffusion_latent_upscale.py`

```py
# 版权声明，说明该文件的版权所有者及其保留的权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 许可证第 2.0 版的规定使用此文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 您不得在未遵守许可证的情况下使用此文件
# you may not use this file except in compliance with the License.
# 可在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议另有约定，软件按“原样”提供
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 无任何形式的保证或条件，无论明示或暗示
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请查看许可证以获取特定语言的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings  # 导入警告模块，用于发出警告信息
from typing import Callable, List, Optional, Union  # 导入类型提示工具

import numpy as np  # 导入 NumPy 库以支持数组操作
import PIL.Image  # 导入 PIL 库用于图像处理
import torch  # 导入 PyTorch 库用于深度学习操作
import torch.nn.functional as F  # 导入 PyTorch 的功能性神经网络模块
from transformers import CLIPTextModel, CLIPTokenizer  # 从 transformers 库导入 CLIP 模型和分词器

from ...image_processor import PipelineImageInput, VaeImageProcessor  # 从本地模块导入图像处理相关类
from ...loaders import FromSingleFileMixin  # 从本地模块导入单文件加载器
from ...models import AutoencoderKL, UNet2DConditionModel  # 从本地模块导入模型
from ...schedulers import EulerDiscreteScheduler  # 从本地模块导入调度器
from ...utils import deprecate, logging  # 从本地模块导入工具函数和日志记录
from ...utils.torch_utils import randn_tensor  # 从本地模块导入随机张量生成函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput, StableDiffusionMixin  # 从管道工具导入相关类


logger = logging.get_logger(__name__)  # 创建一个日志记录器，用于记录信息，禁用 pylint 的无效名称警告


# 预处理函数，用于处理输入图像
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale.preprocess
def preprocess(image):
    # 发出关于该方法已弃用的警告，提示将来版本会移除
    warnings.warn(
        "The preprocess method is deprecated and will be removed in a future version. Please"
        " use VaeImageProcessor.preprocess instead",
        FutureWarning,
    )
    # 如果输入是张量，直接返回
    if isinstance(image, torch.Tensor):
        return image
    # 如果输入是 PIL 图像，将其包装成列表
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    # 检查列表中的第一个元素是否为 PIL 图像
    if isinstance(image[0], PIL.Image.Image):
        # 获取图像的宽度和高度
        w, h = image[0].size
        # 将宽度和高度调整为 64 的整数倍
        w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64

        # 将图像调整为新的大小并转换为数组格式
        image = [np.array(i.resize((w, h)))[None, :] for i in image]
        # 沿第一个轴连接图像数组
        image = np.concatenate(image, axis=0)
        # 归一化图像数据并转换数据类型
        image = np.array(image).astype(np.float32) / 255.0
        # 调整数组维度顺序
        image = image.transpose(0, 3, 1, 2)
        # 将图像数据从 [0, 1] 范围缩放到 [-1, 1] 范围
        image = 2.0 * image - 1.0
        # 将 NumPy 数组转换为 PyTorch 张量
        image = torch.from_numpy(image)
    # 如果列表中的第一个元素是张量，沿着第 0 维度连接它们
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    # 返回处理后的图像
    return image


class StableDiffusionLatentUpscalePipeline(DiffusionPipeline, StableDiffusionMixin, FromSingleFileMixin):
    r"""
    用于将 Stable Diffusion 输出图像分辨率按 2 倍放大的管道。

    该模型继承自 [`DiffusionPipeline`]. 有关所有管道通用方法（下载、保存、在特定设备上运行等）的文档，请查看超类文档
    The pipeline also inherits the following loading methods:
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
    # 定义参数说明
    Args:
        vae ([`AutoencoderKL`]):
            # 用于编码和解码图像的变分自编码器（VAE）模型，处理图像和潜在表示之间的转换
        text_encoder ([`~transformers.CLIPTextModel`]):
            # 冻结的文本编码器，使用 [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) 模型
        tokenizer ([`~transformers.CLIPTokenizer`]):
            # 一个 `CLIPTokenizer` 用于对文本进行分词
        unet ([`UNet2DConditionModel`]):
            # 用于对编码后的图像潜在表示进行去噪的 `UNet2DConditionModel`
        scheduler ([`SchedulerMixin`]):
            # 一个 [`EulerDiscreteScheduler`]，与 `unet` 配合使用以去噪编码后的图像潜在表示
    """

    # 定义模型的计算顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"

    def __init__(
        # 初始化方法，接受多个模型作为参数
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: EulerDiscreteScheduler,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模块，将传入的模型进行存储
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        # 计算 VAE 的缩放因子，基于配置中的块输出通道数量
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # 创建图像处理器，用于处理解码后的图像
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, resample="bicubic")

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline 中复制的方法，用于解码潜在表示
    def decode_latents(self, latents):
        # 设置弃用信息，提示用户该方法将在1.0.0版本中移除
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        # 根据 VAE 的缩放因子调整潜在表示的值
        latents = 1 / self.vae.config.scaling_factor * latents
        # 解码潜在表示，返回图像
        image = self.vae.decode(latents, return_dict=False)[0]
        # 将解码后的图像归一化到[0, 1]范围
        image = (image / 2 + 0.5).clamp(0, 1)
        # 转换图像数据为 float32 格式，以确保与 bfloat16 兼容，并避免显著开销
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        # 返回处理后的图像
        return image
    # 检查输入参数的有效性
    def check_inputs(self, prompt, image, callback_steps):
        # 如果 `prompt` 不是字符串或列表类型，则引发错误
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 如果 `image` 不是张量、PIL图像或列表类型，则引发错误
        if (
            not isinstance(image, torch.Tensor)
            and not isinstance(image, PIL.Image.Image)
            and not isinstance(image, list)
        ):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or `list` but is {type(image)}"
            )

        # 验证如果 `image` 是列表或张量，则 `prompt` 和 `image` 的批量大小应相同
        if isinstance(image, (list, torch.Tensor)):
            # 如果 `prompt` 是字符串，批量大小为1
            if isinstance(prompt, str):
                batch_size = 1
            else:
                # 否则，批量大小为 `prompt` 的长度
                batch_size = len(prompt)
            # 如果 `image` 是列表，获取其批量大小
            if isinstance(image, list):
                image_batch_size = len(image)
            else:
                # 否则，获取 `image` 的第一维度大小（批量大小）
                image_batch_size = image.shape[0] if image.ndim == 4 else 1
            # 如果 `prompt` 和 `image` 的批量大小不匹配，则引发错误
            if batch_size != image_batch_size:
                raise ValueError(
                    f"`prompt` has batch size {batch_size} and `image` has batch size {image_batch_size}."
                    " Please make sure that passed `prompt` matches the batch size of `image`."
                )

        # 检查 `callback_steps` 是否为正整数
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_upscale 中复制的方法
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜在变量的形状
        shape = (batch_size, num_channels_latents, height, width)
        # 如果没有提供 `latents`，则生成随机的潜在变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供的 `latents` 形状不匹配，则引发错误
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将 `latents` 移动到指定设备
            latents = latents.to(device)

        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    # 在计算图中不追踪梯度
    @torch.no_grad()
    # 定义一个可调用的类方法，接收多个参数用于生成图像
        def __call__(
            self,
            # 用户输入的提示文本，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 输入的图像，默认为 None
            image: PipelineImageInput = None,
            # 推理步骤的数量，默认为 75
            num_inference_steps: int = 75,
            # 指导缩放因子，默认为 9.0
            guidance_scale: float = 9.0,
            # 负提示文本，可以是字符串或字符串列表，默认为 None
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 随机数生成器，可以是单个或多个 torch.Generator，默认为 None
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil" (Python Imaging Library)
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 回调函数，用于在推理过程中处理某些操作，默认为 None
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调函数调用的步数间隔，默认为 1
            callback_steps: int = 1,
```