# `.\diffusers\pipelines\kandinsky2_2\pipeline_kandinsky2_2_controlnet_img2img.py`

```py
# 版权声明，表示该代码的版权所有者和保留权利
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可证的规定进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 你必须遵守许可证才能使用此文件
# you may not use this file except in compliance with the License.
# 许可证可以在以下地址获取
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件按“原样”分发
# Unless required by applicable law or agreed to in writing, software
# 分发不提供任何明示或暗示的担保或条件
# distributed under the License is distributed on an "AS IS" BASIS,
# 参见许可证以了解特定的权限和限制
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 从 typing 模块导入类型注解
from typing import Callable, List, Optional, Union

# 导入 numpy 库，通常用于数值计算
import numpy as np
# 导入 PIL.Image 模块，用于图像处理
import PIL.Image
# 导入 torch 库，深度学习框架
import torch
# 从 PIL 导入 Image 模块，用于图像处理
from PIL import Image

# 从相对路径导入 UNet2DConditionModel 和 VQModel 模型
from ...models import UNet2DConditionModel, VQModel
# 从相对路径导入 DDPMScheduler 调度器
from ...schedulers import DDPMScheduler
# 从相对路径导入 logging 实用程序
from ...utils import (
    logging,
)
# 从相对路径导入 randn_tensor 函数
from ...utils.torch_utils import randn_tensor
# 从相对路径导入 DiffusionPipeline 和 ImagePipelineOutput
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 创建一个日志记录器，用于记录当前模块的日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，通常用于说明代码示例
EXAMPLE_DOC_STRING = """
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
``` 
```py 
    # 示例代码，演示如何使用 Kandinsky 和 Transformers 进行图像处理
    Examples:
        ```py
        # 导入 PyTorch 库
        >>> import torch
        # 导入 NumPy 库
        >>> import numpy as np

        # 从 diffusers 导入所需的管道类
        >>> from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
        # 从 transformers 导入管道函数
        >>> from transformers import pipeline
        # 从 diffusers.utils 导入图像加载函数
        >>> from diffusers.utils import load_image


        # 定义生成提示的函数，接受图像和深度估计器作为参数
        >>> def make_hint(image, depth_estimator):
        ...     # 使用深度估计器处理图像，获取深度图
        ...     image = depth_estimator(image)["depth"]
        ...     # 将深度图转换为 NumPy 数组
        ...     image = np.array(image)
        ...     # 在数组的最后一维添加一个新的维度
        ...     image = image[:, :, None]
        ...     # 将深度图复制三次，形成 RGB 格式
        ...     image = np.concatenate([image, image, image], axis=2)
        ...     # 将 NumPy 数组转换为 PyTorch 张量并标准化到 [0, 1]
        ...     detected_map = torch.from_numpy(image).float() / 255.0
        ...     # 重新排列张量的维度，准备为输入
        ...     hint = detected_map.permute(2, 0, 1)
        ...     # 返回处理后的提示张量
        ...     return hint


        # 创建深度估计器管道
        >>> depth_estimator = pipeline("depth-estimation")

        # 从预训练模型加载 Kandinsky V2 Prior 管道
        >>> pipe_prior = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        # 将管道移动到 GPU
        >>> pipe_prior = pipe_prior.to("cuda")

        # 从预训练模型加载 Kandinsky V2 Controlnet 图像到图像管道
        >>> pipe = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-controlnet-depth", torch_dtype=torch.float16
        ... )
        # 将管道移动到 GPU
        >>> pipe = pipe.to("cuda")

        # 从 URL 加载图像并调整大小
        >>> img = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... ).resize((768, 768))


        # 生成图像的提示，扩展维度并转换为半精度格式，移动到 GPU
        >>> hint = make_hint(img, depth_estimator).unsqueeze(0).half().to("cuda")

        # 设置生成的提示文本
        >>> prompt = "A robot, 4k photo"
        # 设置负面提示文本，避免某些特征
        >>> negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

        # 创建一个随机数生成器，设置种子为 43，确保可复现性
        >>> generator = torch.Generator(device="cuda").manual_seed(43)

        # 生成与正面提示对应的图像嵌入
        >>> img_emb = pipe_prior(prompt=prompt, image=img, strength=0.85, generator=generator)
        # 生成与负面提示对应的图像嵌入
        >>> negative_emb = pipe_prior(prompt=negative_prior_prompt, image=img, strength=1, generator=generator)

        # 使用控制管道生成图像，包含多种参数设置
        >>> images = pipe(
        ...     image=img,
        ...     strength=0.5,
        ...     image_embeds=img_emb.image_embeds,
        ...     negative_image_embeds=negative_emb.image_embeds,
        ...     hint=hint,
        ...     num_inference_steps=50,
        ...     generator=generator,
        ...     height=768,
        ...     width=768,
        ... ).images

        # 将生成的第一张图像保存到文件
        >>> images[0].save("robot_cat.png")
# Copied from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2.downscale_height_and_width
def downscale_height_and_width(height, width, scale_factor=8):
    # 计算新的高度，按照比例因子缩小
    new_height = height // scale_factor**2
    # 如果高度不是比例因子的整数倍，向上取整
    if height % scale_factor**2 != 0:
        new_height += 1
    # 计算新的宽度，按照比例因子缩小
    new_width = width // scale_factor**2
    # 如果宽度不是比例因子的整数倍，向上取整
    if width % scale_factor**2 != 0:
        new_width += 1
    # 返回调整后的高度和宽度
    return new_height * scale_factor, new_width * scale_factor


# Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.prepare_image
def prepare_image(pil_image, w=512, h=512):
    # 调整 PIL 图像大小为指定宽度和高度，使用双三次插值法
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    # 将 PIL 图像转换为 RGB 数组，并归一化到 [-1, 1] 范围内的浮点数
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    # 调整数组维度顺序为 [通道, 高度, 宽度]，并转换为 PyTorch 张量
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr).unsqueeze(0)
    # 返回处理后的图像张量
    return image


class KandinskyV22ControlnetImg2ImgPipeline(DiffusionPipeline):
    """
    Pipeline for image-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        scheduler ([`DDIMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
    """

    model_cpu_offload_seq = "unet->movq"

    def __init__(
        self,
        unet: UNet2DConditionModel,
        scheduler: DDPMScheduler,
        movq: VQModel,
    ):
        super().__init__()
        
        # 注册模块到管道中
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        # 计算 MoVQ 缩放因子
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    # Copied from diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img.KandinskyImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # 根据推断步数和强度计算初始时间步长
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算起始时间步，确保不超出范围
        t_start = max(num_inference_steps - init_timestep, 0)
        # 获取调度器的时间步长
        timesteps = self.scheduler.timesteps[t_start:]

        # 返回时间步长列表和有效时间步数
        return timesteps, num_inference_steps - t_start

    # Copied from diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2_img2img.KandinskyV22Img2ImgPipeline.prepare_latents
    # 准备潜在变量，用于图像生成模型
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        # 检查输入的图像类型是否为张量、PIL图像或列表
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            # 如果不是，则抛出值错误，说明类型不匹配
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        # 将图像转换到指定设备和数据类型
        image = image.to(device=device, dtype=dtype)

        # 计算有效的批量大小
        batch_size = batch_size * num_images_per_prompt

        # 如果图像的通道数为4，初始化潜在变量为图像本身
        if image.shape[1] == 4:
            init_latents = image

        else:
            # 如果生成器是列表并且其长度与批量大小不匹配，则抛出值错误
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            # 如果生成器是列表，使用每个生成器编码图像并采样潜在变量
            elif isinstance(generator, list):
                init_latents = [
                    self.movq.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                # 将潜在变量沿指定维度连接
                init_latents = torch.cat(init_latents, dim=0)
            else:
                # 如果生成器不是列表，直接编码图像并采样潜在变量
                init_latents = self.movq.encode(image).latent_dist.sample(generator)

            # 根据配置缩放因子调整潜在变量
            init_latents = self.movq.config.scaling_factor * init_latents

        # 再次连接潜在变量，以确保其形状正确
        init_latents = torch.cat([init_latents], dim=0)

        # 获取潜在变量的形状
        shape = init_latents.shape
        # 生成与潜在变量形状相同的噪声张量
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 将噪声添加到潜在变量中，以获得最终的潜在变量
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        # 将潜在变量赋值给另一个变量
        latents = init_latents

        # 返回最终的潜在变量
        return latents

    # 装饰器，表示此函数在执行时不需要梯度计算
    @torch.no_grad()
    def __call__(
        # 定义输入参数，包括图像嵌入、图像及其其他参数
        image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
        negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],
        hint: torch.Tensor,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 100,
        guidance_scale: float = 4.0,
        strength: float = 0.3,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        return_dict: bool = True,
```