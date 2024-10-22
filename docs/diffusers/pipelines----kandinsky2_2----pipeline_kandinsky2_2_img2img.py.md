# `.\diffusers\pipelines\kandinsky2_2\pipeline_kandinsky2_2_img2img.py`

```py
# 版权声明，表明此文件的所有权及使用许可
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 在 Apache 许可证 2.0 版本下许可使用本文件
# Licensed under the Apache License, Version 2.0 (the "License");
# 除非符合该许可证，否则不得使用此文件
# you may not use this file except in compliance with the License.
# 可以在以下网址获取许可证副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，软件在许可证下以“原样”提供
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的明示或暗示的保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请查看许可证以了解有关权限和限制的具体条款
# See the License for the specific language governing permissions and
# limitations under the License.

# 从 typing 模块导入类型注解
from typing import Callable, Dict, List, Optional, Union

# 导入 numpy 库用于数组操作
import numpy as np
# 导入 PIL 库中的 Image 模块用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 PIL 导入 Image 模块
from PIL import Image

# 从本地模型导入 UNet2DConditionModel 和 VQModel
from ...models import UNet2DConditionModel, VQModel
# 从调度器导入 DDPMScheduler
from ...schedulers import DDPMScheduler
# 从工具模块导入 deprecate 和 logging
from ...utils import deprecate, logging
# 从 torch_utils 导入 randn_tensor 函数
from ...utils.torch_utils import randn_tensor
# 从 pipeline_utils 导入 DiffusionPipeline 和 ImagePipelineOutput
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

# 创建一个记录器实例，用于记录日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，包含使用示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22Img2ImgPipeline, KandinskyV22PriorPipeline
        >>> from diffusers.utils import load_image
        >>> import torch

        >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior.to("cuda")

        >>> prompt = "A red cartoon frog, 4k"
        >>> image_emb, zero_image_emb = pipe_prior(prompt, return_dict=False)

        >>> pipe = KandinskyV22Img2ImgPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")

        >>> init_image = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/frog.png"
        ... )

        >>> image = pipe(
        ...     image=init_image,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=100,
        ...     strength=0.2,
        ... ).images

        >>> image[0].save("red_frog.png")
        ```py
"""

# 从 diffusers.pipelines.kandinsky2_2.pipeline_kandinsky2_2 导入下采样高度和宽度的函数
def downscale_height_and_width(height, width, scale_factor=8):
    # 根据比例因子计算新的高度，进行整除
    new_height = height // scale_factor**2
    # 如果高度不能被比例因子平方整除，增加高度
    if height % scale_factor**2 != 0:
        new_height += 1
    # 根据比例因子计算新的宽度，进行整除
    new_width = width // scale_factor**2
    # 如果宽度不能被比例因子平方整除，增加宽度
    if width % scale_factor**2 != 0:
        new_width += 1
    # 返回按比例因子调整后的高度和宽度
    return new_height * scale_factor, new_width * scale_factor

# 从 diffusers.pipelines.kandinsky.pipeline_kandinsky_img2img 导入准备图像的函数
def prepare_image(pil_image, w=512, h=512):
    # 将输入的 PIL 图像调整到指定的宽度和高度，使用 BICUBIC 插值
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    # 将 PIL 图像转换为 RGB 格式并转为 NumPy 数组
        arr = np.array(pil_image.convert("RGB"))
        # 将数组数据类型转换为 float32，并归一化到 [-1, 1] 范围
        arr = arr.astype(np.float32) / 127.5 - 1
        # 转置数组，以改变维度顺序从 (高度, 宽度, 通道) 到 (通道, 高度, 宽度)
        arr = np.transpose(arr, [2, 0, 1])
        # 将 NumPy 数组转换为 PyTorch 张量，并在第一个维度上增加一个维度
        image = torch.from_numpy(arr).unsqueeze(0)
        # 返回处理后的图像张量
        return image
# 定义 Kandinsky 图像到图像生成的管道，继承自 DiffusionPipeline
class KandinskyV22Img2ImgPipeline(DiffusionPipeline):
    """
    使用 Kandinsky 进行图像到图像生成的管道

    此模型继承自 [`DiffusionPipeline`]。查看父类文档以了解库为所有管道实现的通用方法
    （例如下载、保存、在特定设备上运行等）。

    参数：
        scheduler ([`DDIMScheduler`]):
            与 `unet` 结合使用以生成图像潜变量的调度器。
        unet ([`UNet2DConditionModel`]):
            用于去噪图像嵌入的条件 U-Net 结构。
        movq ([`VQModel`]):
            MoVQ 解码器，用于从潜变量生成图像。
    """

    # 定义模型中 CPU 卸载的顺序
    model_cpu_offload_seq = "unet->movq"
    # 定义需要回调的张量输入
    _callback_tensor_inputs = ["latents", "image_embeds", "negative_image_embeds"]

    # 初始化方法，设置 unet、scheduler 和 movq
    def __init__(
        self,
        unet: UNet2DConditionModel,  # 条件 U-Net 模型
        scheduler: DDPMScheduler,    # DDPM 调度器
        movq: VQModel,               # VQ 解码器模型
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册模块以便在管道中使用
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )
        # 计算 movq 的缩放因子
        self.movq_scale_factor = 2 ** (len(self.movq.config.block_out_channels) - 1)

    # 从 KandinskyImg2ImgPipeline 获取时间步的复制方法
    def get_timesteps(self, num_inference_steps, strength, device):
        # 使用 init_timestep 获取原始时间步
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算起始时间步，确保不小于 0
        t_start = max(num_inference_steps - init_timestep, 0)
        # 从调度器中获取相应的时间步
        timesteps = self.scheduler.timesteps[t_start:]

        # 返回计算出的时间步和剩余的推理步骤数
        return timesteps, num_inference_steps - t_start
    # 准备潜在变量，输入图像及其他参数进行处理
    def prepare_latents(self, image, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        # 检查输入图像是否为有效类型
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            # 如果不符合，抛出类型错误
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )
    
        # 将图像转换为指定设备和数据类型
        image = image.to(device=device, dtype=dtype)
    
        # 计算有效批量大小
        batch_size = batch_size * num_images_per_prompt
    
        # 如果图像有四个通道，初始化潜在变量为图像本身
        if image.shape[1] == 4:
            init_latents = image
    
        else:
            # 检查生成器列表长度是否与批量大小匹配
            if isinstance(generator, list) and len(generator) != batch_size:
                # 如果不匹配，抛出错误
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            elif isinstance(generator, list):
                # 如果是生成器列表，逐个图像编码并采样
                init_latents = [
                    self.movq.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
                ]
                # 将所有潜在变量合并
                init_latents = torch.cat(init_latents, dim=0)
            else:
                # 否则直接编码并采样
                init_latents = self.movq.encode(image).latent_dist.sample(generator)
    
            # 对潜在变量进行缩放
            init_latents = self.movq.config.scaling_factor * init_latents
    
        # 将潜在变量维度扩展为批量维度
        init_latents = torch.cat([init_latents], dim=0)
    
        # 获取潜在变量的形状
        shape = init_latents.shape
        # 生成噪声张量
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    
        # 添加噪声到潜在变量中
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
    
        # 设置最终的潜在变量
        latents = init_latents
    
        # 返回处理后的潜在变量
        return latents
    
        # 获取引导比例的属性
        @property
        def guidance_scale(self):
            return self._guidance_scale
    
        # 检查是否使用无分类器引导
        @property
        def do_classifier_free_guidance(self):
            return self._guidance_scale > 1
    
        # 获取时间步数的属性
        @property
        def num_timesteps(self):
            return self._num_timesteps
    
        # 在无梯度上下文中调用
        @torch.no_grad()
        def __call__(
            self,
            image_embeds: Union[torch.Tensor, List[torch.Tensor]],
            image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
            negative_image_embeds: Union[torch.Tensor, List[torch.Tensor]],
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 100,
            guidance_scale: float = 4.0,
            strength: float = 0.3,
            num_images_per_prompt: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
```