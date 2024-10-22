# `.\diffusers\pipelines\kandinsky2_2\pipeline_kandinsky2_2_combined.py`

```py
# 版权信息，注明该代码的所有权归 HuggingFace 团队所有
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# 根据 Apache 2.0 许可协议进行许可
# Licensed under the Apache License, Version 2.0 (the "License");
# 除非遵守该许可，否则不得使用此文件
# you may not use this file except in compliance with the License.
# 你可以在以下网址获取许可的副本
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议，否则软件以“原样”方式分发
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# 不提供任何形式的保证或条件，明示或暗示
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 请参见许可证以了解特定语言的权限和限制
# See the License for the specific language governing permissions and
# limitations under the License.

# 导入类型相关的类，以支持类型注解
from typing import Callable, Dict, List, Optional, Union

# 导入 PIL 库中的 Image 模块，用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 transformers 库中导入 CLIP 相关模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

# 从本地模块导入各种模型和调度器
from ...models import PriorTransformer, UNet2DConditionModel, VQModel
from ...schedulers import DDPMScheduler, UnCLIPScheduler
from ...utils import deprecate, logging, replace_example_docstring
from ..pipeline_utils import DiffusionPipeline
from .pipeline_kandinsky2_2 import KandinskyV22Pipeline
from .pipeline_kandinsky2_2_img2img import KandinskyV22Img2ImgPipeline
from .pipeline_kandinsky2_2_inpainting import KandinskyV22InpaintPipeline
from .pipeline_kandinsky2_2_prior import KandinskyV22PriorPipeline

# 创建日志记录器，用于记录模块中的日志信息
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义文本到图像转换的示例文档字符串
TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        # 从 diffusers 库中导入自动文本到图像管道
        from diffusers import AutoPipelineForText2Image
        # 导入 PyTorch 库
        import torch

        # 加载预训练的文本到图像管道，使用半精度浮点数
        pipe = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        )
        # 启用模型的 CPU 卸载，以减少内存占用
        pipe.enable_model_cpu_offload()

        # 定义要生成图像的提示语
        prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"

        # 生成图像，指定推理步骤数量，并获取生成的第一张图像
        image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        ```py
"""

# 定义图像到图像转换的示例文档字符串
IMAGE2IMAGE_EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        # 从 diffusers 库中导入自动图像到图像管道
        from diffusers import AutoPipelineForImage2Image
        # 导入 PyTorch 库
        import torch
        # 导入请求库用于下载图像
        import requests
        # 从 io 库中导入 BytesIO，用于处理字节流
        from io import BytesIO
        # 导入 PIL 库中的 Image 模块
        from PIL import Image
        # 导入 os 库
        import os

        # 加载预训练的图像到图像管道，使用半精度浮点数
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
        )
        # 启用模型的 CPU 卸载，以减少内存占用
        pipe.enable_model_cpu_offload()

        # 定义要生成图像的提示语
        prompt = "A fantasy landscape, Cinematic lighting"
        # 定义负面提示语，用于控制生成的图像质量
        negative_prompt = "low quality, bad quality"

        # 定义要下载的图像 URL
        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        # 发起 GET 请求以获取图像
        response = requests.get(url)
        # 打开获取的图像内容，并将其转换为 RGB 模式
        image = Image.open(BytesIO(response.content)).convert("RGB")
        # 调整图像大小，以适应后续处理
        image.thumbnail((768, 768))

        # 使用管道生成新图像，传入原始图像和提示语，获取生成的第一张图像
        image = pipe(prompt=prompt, image=original_image, num_inference_steps=25).images[0]
        ```py
"""

# 定义图像修复的示例文档字符串
INPAINT_EXAMPLE_DOC_STRING = """
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
    # 示例代码展示如何使用 Diffusers 库进行图像修复
        Examples:
            ```py
            # 从 diffusers 库导入 AutoPipelineForInpainting 类
            from diffusers import AutoPipelineForInpainting
            # 从 diffusers.utils 导入 load_image 函数
            from diffusers.utils import load_image
            # 导入 PyTorch 库
            import torch
            # 导入 NumPy 库
            import numpy as np
    
            # 从预训练模型加载图像修复管道，指定数据类型为 float16
            pipe = AutoPipelineForInpainting.from_pretrained(
                "kandinsky-community/kandinsky-2-2-decoder-inpaint", torch_dtype=torch.float16
            )
            # 启用模型的 CPU 内存卸载功能，以节省显存
            pipe.enable_model_cpu_offload()
    
            # 定义图像修复的提示语
            prompt = "A fantasy landscape, Cinematic lighting"
            # 定义负面提示，以避免生成低质量图像
            negative_prompt = "low quality, bad quality"
    
            # 加载原始图像，使用给定的 URL
            original_image = load_image(
                "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
            )
    
            # 创建一个全零的掩码，大小为 (768, 768)，数据类型为 float32
            mask = np.zeros((768, 768), dtype=np.float32)
            # 在猫的头部上方的区域设置掩码为 1，以进行遮挡
            mask[:250, 250:-250] = 1
    
            # 使用管道进行图像修复，传入提示、原始图像和掩码，指定推理步骤数为 25，获取生成的图像
            image = pipe(prompt=prompt, image=original_image, mask_image=mask, num_inference_steps=25).images[0]
            ```py 
# 定义一个结合管道类，用于基于 Kandinsky 的文本到图像生成
class KandinskyV22CombinedPipeline(DiffusionPipeline):
    # 文档字符串，描述该类的功能和参数
    """
    Combined Pipeline for text-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
        prior_prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        prior_tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        prior_image_processor ([`CLIPImageProcessor`]):
            A image_processor to be used to preprocess image from clip.
    """
    
    # 定义模型的 CPU 卸载顺序，指定各部分的执行顺序
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->unet->movq"
    # 表示需要加载连接的管道
    _load_connected_pipes = True
    # 指定从 CPU 卸载时要排除的组件
    _exclude_from_cpu_offload = ["prior_prior"]

    # 初始化方法，接受多个模型作为参数
    def __init__(
        self,
        unet: UNet2DConditionModel,  # 条件 U-Net 模型，用于图像去噪
        scheduler: DDPMScheduler,  # 调度器，用于生成图像潜在特征
        movq: VQModel,  # MoVQ 解码器，从潜在特征生成图像
        prior_prior: PriorTransformer,  # 用于近似图像嵌入的先验转换器
        prior_image_encoder: CLIPVisionModelWithProjection,  # 冻结的图像编码器
        prior_text_encoder: CLIPTextModelWithProjection,  # 冻结的文本编码器
        prior_tokenizer: CLIPTokenizer,  # CLIP 的分词器
        prior_scheduler: UnCLIPScheduler,  # 生成图像嵌入的调度器
        prior_image_processor: CLIPImageProcessor,  # 用于预处理图像的图像处理器
    # 初始化父类
        ):
            super().__init__()
    
            # 注册各个模块，以便于后续使用
            self.register_modules(
                unet=unet,
                scheduler=scheduler,
                movq=movq,
                prior_prior=prior_prior,
                prior_image_encoder=prior_image_encoder,
                prior_text_encoder=prior_text_encoder,
                prior_tokenizer=prior_tokenizer,
                prior_scheduler=prior_scheduler,
                prior_image_processor=prior_image_processor,
            )
            # 创建先前处理管道，用于图像和文本处理
            self.prior_pipe = KandinskyV22PriorPipeline(
                prior=prior_prior,
                image_encoder=prior_image_encoder,
                text_encoder=prior_text_encoder,
                tokenizer=prior_tokenizer,
                scheduler=prior_scheduler,
                image_processor=prior_image_processor,
            )
            # 创建解码管道，负责生成最终输出
            self.decoder_pipe = KandinskyV22Pipeline(
                unet=unet,
                scheduler=scheduler,
                movq=movq,
            )
    
        # 启用高效内存注意力机制
        def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
            # 调用解码管道的方法，启用内存高效注意力
            self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)
    
        # 启用顺序 CPU 卸载
        def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
            r"""
            将所有模型卸载到 CPU，从而显著减少内存使用。当调用时，unet、text_encoder、vae 和安全检查器的状态字典保存到 CPU，然后移至
            `torch.device('meta')，并在特定子模块调用其 `forward` 方法时加载到 GPU。
            注意，卸载是针对子模块进行的。内存节省比 `enable_model_cpu_offload` 高，但性能较低。
            """
            # 启用先前处理管道的顺序 CPU 卸载
            self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
            # 启用解码管道的顺序 CPU 卸载
            self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
    
        # 进度条功能
        def progress_bar(self, iterable=None, total=None):
            # 调用先前处理管道的进度条功能
            self.prior_pipe.progress_bar(iterable=iterable, total=total)
            # 调用解码管道的进度条功能
            self.decoder_pipe.progress_bar(iterable=iterable, total=total)
            # 启用解码管道的模型 CPU 卸载
            self.decoder_pipe.enable_model_cpu_offload()
    
        # 设置进度条的配置
        def set_progress_bar_config(self, **kwargs):
            # 为先前处理管道设置进度条配置
            self.prior_pipe.set_progress_bar_config(**kwargs)
            # 为解码管道设置进度条配置
            self.decoder_pipe.set_progress_bar_config(**kwargs)
    
        # 无梯度计算装饰器，避免计算梯度
        @torch.no_grad()
        # 替换示例文档字符串
        @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法，接受多个参数进行处理
        def __call__(
            self,
            # 输入的提示，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 可选的负面提示，也可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 推理步骤的数量，默认为100
            num_inference_steps: int = 100,
            # 指导比例，默认为4.0
            guidance_scale: float = 4.0,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: int = 1,
            # 输出图像的高度，默认为512
            height: int = 512,
            # 输出图像的宽度，默认为512
            width: int = 512,
            # 先前指导比例，默认为4.0
            prior_guidance_scale: float = 4.0,
            # 先前推理步骤的数量，默认为25
            prior_num_inference_steps: int = 25,
            # 随机数生成器，可选，支持单个或多个生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜变量，默认为None
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为"pil"
            output_type: Optional[str] = "pil",
            # 可选的回调函数，用于处理推理过程中的步骤
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调函数触发的步骤间隔，默认为1
            callback_steps: int = 1,
            # 返回结果的字典形式，默认为True
            return_dict: bool = True,
            # 可选的先前回调函数，在步骤结束时调用
            prior_callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 用于先前回调的张量输入，默认为["latents"]
            prior_callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 可选的回调函数，在步骤结束时调用
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 用于回调的张量输入，默认为["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
# 定义一个结合图像到图像生成的管道类 KandinskyV22Img2ImgCombinedPipeline，继承自 DiffusionPipeline
class KandinskyV22Img2ImgCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for image-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.
        prior_prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        prior_tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        prior_image_processor ([`CLIPImageProcessor`]):
            A image_processor to be used to preprocess image from clip.
    """

    # 定义 CPU 卸载顺序的模型组件
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->unet->movq"
    # 指定是否加载连接的管道
    _load_connected_pipes = True
    # 指定从 CPU 卸载中排除的组件
    _exclude_from_cpu_offload = ["prior_prior"]

    # 初始化方法，接受多个模型组件作为参数
    def __init__(
        # 条件 U-Net 模型，用于图像嵌入去噪
        self,
        unet: UNet2DConditionModel,
        # DDPMScheduler 调度器
        scheduler: DDPMScheduler,
        # MoVQ 解码器
        movq: VQModel,
        # 用于图像嵌入的 unCLIP 先验
        prior_prior: PriorTransformer,
        # 冻结的图像编码器
        prior_image_encoder: CLIPVisionModelWithProjection,
        # 冻结的文本编码器
        prior_text_encoder: CLIPTextModelWithProjection,
        # CLIP 令牌化器
        prior_tokenizer: CLIPTokenizer,
        # 先验的调度器
        prior_scheduler: UnCLIPScheduler,
        # 图像预处理器
        prior_image_processor: CLIPImageProcessor,
    # 初始化父类
        ):
            super().__init__()
    
            # 注册多个模块以供后续使用
            self.register_modules(
                unet=unet,  # UNet模型
                scheduler=scheduler,  # 调度器
                movq=movq,  # MovQ模型
                prior_prior=prior_prior,  # 先验模型
                prior_image_encoder=prior_image_encoder,  # 图像编码器
                prior_text_encoder=prior_text_encoder,  # 文本编码器
                prior_tokenizer=prior_tokenizer,  # 标记器
                prior_scheduler=prior_scheduler,  # 先验调度器
                prior_image_processor=prior_image_processor,  # 图像处理器
            )
            # 创建先验管道实例
            self.prior_pipe = KandinskyV22PriorPipeline(
                prior=prior_prior,  # 先验模型
                image_encoder=prior_image_encoder,  # 图像编码器
                text_encoder=prior_text_encoder,  # 文本编码器
                tokenizer=prior_tokenizer,  # 标记器
                scheduler=prior_scheduler,  # 先验调度器
                image_processor=prior_image_processor,  # 图像处理器
            )
            # 创建图像到图像的转换管道实例
            self.decoder_pipe = KandinskyV22Img2ImgPipeline(
                unet=unet,  # UNet模型
                scheduler=scheduler,  # 调度器
                movq=movq,  # MovQ模型
            )
    
        # 启用高效的 xformers 内存注意力机制
        def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
            self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)
    
        # 启用模型 CPU 卸载
        def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
            r"""
            使用 accelerate 将所有模型卸载到 CPU，减少内存使用，性能影响低。与 `enable_sequential_cpu_offload` 相比，
            此方法在调用 `forward` 方法时将整个模型一次性移动到 GPU，并在下一个模型运行前保持在 GPU。
            内存节省较少，但性能因 UNet 的迭代执行而更好。
            """
            # 启用先验管道的 CPU 卸载
            self.prior_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
            # 启用解码管道的 CPU 卸载
            self.decoder_pipe.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
    
        # 启用顺序 CPU 卸载
        def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
            r"""
            使用 accelerate 将所有模型卸载到 CPU，显著减少内存使用。调用时，unet、text_encoder、vae 和安全检查器
            的状态字典保存到 CPU，然后移动到 `torch.device('meta')`，仅在其特定子模块调用 `forward` 方法时加载到 GPU。
            注意卸载是按子模块进行的。内存节省高于 `enable_model_cpu_offload`，但性能较低。
            """
            # 启用先验管道的顺序 CPU 卸载
            self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
            # 启用解码管道的顺序 CPU 卸载
            self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
    
        # 显示进度条
        def progress_bar(self, iterable=None, total=None):
            # 在先验管道中显示进度条
            self.prior_pipe.progress_bar(iterable=iterable, total=total)
            # 在解码管道中显示进度条
            self.decoder_pipe.progress_bar(iterable=iterable, total=total)
            # 启用解码管道的模型 CPU 卸载
            self.decoder_pipe.enable_model_cpu_offload()
    # 设置进度条的配置，允许通过关键字参数自定义
    def set_progress_bar_config(self, **kwargs):
        # 在优先处理管道中设置进度条配置，传入关键字参数
        self.prior_pipe.set_progress_bar_config(**kwargs)
        # 在解码器管道中设置进度条配置，传入关键字参数
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    # 在不计算梯度的情况下进行操作，节省内存
    @torch.no_grad()
    # 用于替换示例文档字符串的装饰器
    @replace_example_docstring(IMAGE2IMAGE_EXAMPLE_DOC_STRING)
    # 定义调用方法，支持多种输入类型和参数配置
    def __call__(
        # 提示文本，可以是单个字符串或字符串列表
        prompt: Union[str, List[str]],
        # 输入图像，可以是张量或 PIL 图像，也支持列表
        image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
        # 可选的负提示文本，用于限制生成
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 推理步骤数量，默认为 100
        num_inference_steps: int = 100,
        # 指导比例，控制生成的多样性，默认为 4.0
        guidance_scale: float = 4.0,
        # 强度参数，控制图像变换的强烈程度，默认为 0.3
        strength: float = 0.3,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: int = 1,
        # 生成图像的高度，默认为 512
        height: int = 512,
        # 生成图像的宽度，默认为 512
        width: int = 512,
        # 优先指导比例，默认为 4.0
        prior_guidance_scale: float = 4.0,
        # 优先推理步骤数量，默认为 25
        prior_num_inference_steps: int = 25,
        # 随机数生成器，支持单个或多个生成器
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 预设的潜在变量，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 输出类型，默认为 "pil"（PIL 格式）
        output_type: Optional[str] = "pil",
        # 回调函数，用于处理生成过程中的步骤
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调执行的步骤频率，默认为 1
        callback_steps: int = 1,
        # 是否返回字典格式的结果，默认为 True
        return_dict: bool = True,
        # 优先回调函数在步骤结束时执行
        prior_callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 在步骤结束时传入的优先回调的张量输入，默认为 ["latents"]
        prior_callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # 回调函数在步骤结束时执行
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # 在步骤结束时传入的回调的张量输入，默认为 ["latents"]
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
# 定义一个名为 KandinskyV22InpaintCombinedPipeline 的类，继承自 DiffusionPipeline
class KandinskyV22InpaintCombinedPipeline(DiffusionPipeline):
    """
    使用 Kandinsky 进行图像修复生成的组合管道

    该模型继承自 [`DiffusionPipeline`]。有关库为所有管道实现的通用方法（例如下载或保存、在特定设备上运行等），请查看父类文档。

    Args:
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            要与 `unet` 结合使用的调度器，用于生成图像潜在表示。
        unet ([`UNet2DConditionModel`]):
            条件 U-Net 结构，用于去噪图像嵌入。
        movq ([`VQModel`]):
            MoVQ 解码器，从潜在表示生成图像。
        prior_prior ([`PriorTransformer`]):
            经典的 unCLIP 先验，用于近似文本嵌入的图像嵌入。
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            冻结的图像编码器。
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            冻结的文本编码器。
        prior_tokenizer (`CLIPTokenizer`):
            该类的分词器
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)。
        prior_scheduler ([`UnCLIPScheduler`]):
            要与 `prior` 结合使用的调度器，用于生成图像嵌入。
        prior_image_processor ([`CLIPImageProcessor`]):
            用于预处理图像的图像处理器。
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->unet->movq"
    # 指定是否加载连接的管道
    _load_connected_pipes = True
    # 指定要排除在 CPU 卸载之外的组件
    _exclude_from_cpu_offload = ["prior_prior"]

    # 初始化方法，定义管道的参数
    def __init__(
        # 条件 U-Net 模型
        self,
        unet: UNet2DConditionModel,
        # 扩散调度器
        scheduler: DDPMScheduler,
        # MoVQ 解码器
        movq: VQModel,
        # 先验变换器
        prior_prior: PriorTransformer,
        # 图像编码器
        prior_image_encoder: CLIPVisionModelWithProjection,
        # 文本编码器
        prior_text_encoder: CLIPTextModelWithProjection,
        # 分词器
        prior_tokenizer: CLIPTokenizer,
        # 先验调度器
        prior_scheduler: UnCLIPScheduler,
        # 图像处理器
        prior_image_processor: CLIPImageProcessor,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册各个模块，包括 UNet、调度器等
        self.register_modules(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
            prior_prior=prior_prior,
            prior_image_encoder=prior_image_encoder,
            prior_text_encoder=prior_text_encoder,
            prior_tokenizer=prior_tokenizer,
            prior_scheduler=prior_scheduler,
            prior_image_processor=prior_image_processor,
        )
        # 创建 KandinskyV22PriorPipeline 实例，传入必要的组件
        self.prior_pipe = KandinskyV22PriorPipeline(
            prior=prior_prior,
            image_encoder=prior_image_encoder,
            text_encoder=prior_text_encoder,
            tokenizer=prior_tokenizer,
            scheduler=prior_scheduler,
            image_processor=prior_image_processor,
        )
        # 创建 KandinskyV22InpaintPipeline 实例，传入 UNet 和调度器
        self.decoder_pipe = KandinskyV22InpaintPipeline(
            unet=unet,
            scheduler=scheduler,
            movq=movq,
        )

    # 启用 xformers 的内存高效注意力机制
    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        # 调用解码管道的方法启用内存高效注意力
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    # 启用顺序 CPU 卸载，减少内存使用
    def enable_sequential_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        使用 accelerate 将所有模型卸载到 CPU，显著减少内存使用。调用时，unet、
        text_encoder、vae 和安全检查器的状态字典被保存到 CPU，然后移动到
        `torch.device('meta')，仅在特定子模块调用其 `forward` 方法时加载到 GPU。
        注意，卸载是基于子模块的。内存节省大于 `enable_model_cpu_offload`，但性能较低。
        """
        # 在优先管道上启用顺序 CPU 卸载
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)
        # 在解码管道上启用顺序 CPU 卸载
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id, device=device)

    # 显示进度条
    def progress_bar(self, iterable=None, total=None):
        # 在优先管道上显示进度条
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        # 在解码管道上显示进度条
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)
        # 启用解码管道的模型 CPU 卸载
        self.decoder_pipe.enable_model_cpu_offload()

    # 设置进度条配置
    def set_progress_bar_config(self, **kwargs):
        # 在优先管道上设置进度条配置
        self.prior_pipe.set_progress_bar_config(**kwargs)
        # 在解码管道上设置进度条配置
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    # 使用装饰器以禁用梯度计算
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(INPAINT_EXAMPLE_DOC_STRING)
    # 定义可调用对象的 __call__ 方法，允许实例像函数一样被调用
        def __call__(
            # 输入提示，可以是字符串或字符串列表
            self,
            prompt: Union[str, List[str]],
            # 输入图像，可以是张量、PIL 图像或它们的列表
            image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
            # 掩膜图像，可以是张量、PIL 图像或它们的列表
            mask_image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
            # 可选的负提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 推理步骤的数量，默认为 100
            num_inference_steps: int = 100,
            # 指导比例，默认为 4.0
            guidance_scale: float = 4.0,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 输出图像的高度，默认为 512
            height: int = 512,
            # 输出图像的宽度，默认为 512
            width: int = 512,
            # 先验指导比例，默认为 4.0
            prior_guidance_scale: float = 4.0,
            # 先验推理步骤的数量，默认为 25
            prior_num_inference_steps: int = 25,
            # 随机数生成器，可选，可以是张量或生成器的列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量，默认为 None
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 先验回调函数，可选，接收步骤和状态的回调
            prior_callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 先验回调函数的输入张量名称列表，默认为 ["latents"]
            prior_callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 回调函数，可选，接收步骤和状态的回调
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 回调函数的输入张量名称列表，默认为 ["latents"]
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            # 其他可选参数
            **kwargs,
```