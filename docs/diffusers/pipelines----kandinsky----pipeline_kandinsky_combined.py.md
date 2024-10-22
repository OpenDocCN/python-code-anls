# `.\diffusers\pipelines\kandinsky\pipeline_kandinsky_combined.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（"许可证"）许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件在许可证下分发时以“原样”基础提供，
# 不提供任何形式的保证或条件，无论是明示或暗示的。
# 请参阅许可证以获取管理权限和
# 限制的具体语言。
from typing import Callable, List, Optional, Union  # 从 typing 模块导入类型注解功能

import PIL.Image  # 导入 PIL.Image 模块以处理图像
import torch  # 导入 PyTorch 库用于深度学习
from transformers import (  # 从 transformers 库导入多个模型和处理器
    CLIPImageProcessor,  # 导入 CLIP 图像处理器
    CLIPTextModelWithProjection,  # 导入具有投影的 CLIP 文本模型
    CLIPTokenizer,  # 导入 CLIP 令牌化工具
    CLIPVisionModelWithProjection,  # 导入具有投影的 CLIP 视觉模型
    XLMRobertaTokenizer,  # 导入 XLM-Roberta 令牌化工具
)

from ...models import PriorTransformer, UNet2DConditionModel, VQModel  # 从相对路径导入模型
from ...schedulers import DDIMScheduler, DDPMScheduler, UnCLIPScheduler  # 导入不同的调度器
from ...utils import (  # 从工具模块导入特定功能
    replace_example_docstring,  # 导入替换示例文档字符串的功能
)
from ..pipeline_utils import DiffusionPipeline  # 从上级路径导入扩散管道工具
from .pipeline_kandinsky import KandinskyPipeline  # 导入 Kandinsky 管道
from .pipeline_kandinsky_img2img import KandinskyImg2ImgPipeline  # 导入 Kandinsky 图像到图像管道
from .pipeline_kandinsky_inpaint import KandinskyInpaintPipeline  # 导入 Kandinsky 修复管道
from .pipeline_kandinsky_prior import KandinskyPriorPipeline  # 导入 Kandinsky 先验管道
from .text_encoder import MultilingualCLIP  # 导入多语言 CLIP 文本编码器

# 定义文本到图像的示例文档字符串
TEXT2IMAGE_EXAMPLE_DOC_STRING = """
    示例：
        ```py
        from diffusers import AutoPipelineForText2Image  # 导入自动文本到图像管道
        import torch  # 导入 PyTorch 库

        # 从预训练模型加载管道
        pipe = AutoPipelineForText2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16
        )
        # 启用模型的 CPU 卸载功能
        pipe.enable_model_cpu_offload()

        # 定义生成图像的提示语
        prompt = "A lion in galaxies, spirals, nebulae, stars, smoke, iridescent, intricate detail, octane render, 8k"

        # 生成图像并获取第一张图像
        image = pipe(prompt=prompt, num_inference_steps=25).images[0]
        ```py
"""

# 定义图像到图像的示例文档字符串
IMAGE2IMAGE_EXAMPLE_DOC_STRING = """
    示例：
        ```py
        from diffusers import AutoPipelineForImage2Image  # 导入自动图像到图像管道
        import torch  # 导入 PyTorch 库
        import requests  # 导入请求库用于获取图像
        from io import BytesIO  # 从字节流中读取数据
        from PIL import Image  # 导入 PIL 图像处理库
        import os  # 导入操作系统接口库

        # 从预训练模型加载管道
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16
        )
        # 启用模型的 CPU 卸载功能
        pipe.enable_model_cpu_offload()

        # 定义生成图像的提示语和负面提示语
        prompt = "A fantasy landscape, Cinematic lighting"
        negative_prompt = "low quality, bad quality"

        # 图像 URL
        url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        # 发送请求获取图像
        response = requests.get(url)
        # 打开图像并转换为 RGB 格式
        image = Image.open(BytesIO(response.content)).convert("RGB")
        # 调整图像大小
        image.thumbnail((768, 768))

        # 生成新图像并获取第一张图像
        image = pipe(prompt=prompt, image=original_image, num_inference_steps=25).images[0]
        ```py
"""

# 定义修复的示例文档字符串
INPAINT_EXAMPLE_DOC_STRING = """
``` 
```py  # 结束修复示例文档字符串
    # 示例代码块
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

        # 从预训练模型加载 AutoPipelineForInpainting 对象，指定数据类型为 float16
        pipe = AutoPipelineForInpainting.from_pretrained(
            "kandinsky-community/kandinsky-2-1-inpaint", torch_dtype=torch.float16
        )
        # 启用模型的 CPU 卸载功能，以节省内存
        pipe.enable_model_cpu_offload()

        # 定义提示词，描述要生成的图像内容
        prompt = "A fantasy landscape, Cinematic lighting"
        # 定义负面提示词，用于限制生成内容的质量
        negative_prompt = "low quality, bad quality"

        # 从指定 URL 加载原始图像
        original_image = load_image(
            "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main" "/kandinsky/cat.png"
        )

        # 创建一个全为零的掩码数组，大小为 768x768，数据类型为 float32
        mask = np.zeros((768, 768), dtype=np.float32)
        # 在猫的头部上方遮罩区域
        mask[:250, 250:-250] = 1

        # 使用管道生成新图像，输入提示词、原始图像和掩码，设定推理步骤数量为 25，并提取生成的第一张图像
        image = pipe(prompt=prompt, image=original_image, mask_image=mask, num_inference_steps=25).images[0]
        ``` 
"""
# 类定义：KandinskyCombinedPipeline，继承自 DiffusionPipeline，用于文本到图像生成
class KandinskyCombinedPipeline(DiffusionPipeline):
    """
    # 文档字符串：描述使用Kandinsky进行文本到图像生成的组合管道

    # 文档字符串：说明此模型继承自 [`DiffusionPipeline`]，并提到其通用方法的文档（如下载、保存、在特定设备上运行等）

    # 文档字符串：模型构造函数的参数说明
        # text_encoder：被冻结的文本编码器，类型为 [`MultilingualCLIP`]。
        # tokenizer：类的分词器，类型为 [`XLMRobertaTokenizer`]。
        # scheduler：用于与 `unet` 结合生成图像潜变量的调度器，类型为 `Union[`DDIMScheduler`,`DDPMScheduler`]`。
        # unet：条件U-Net架构，用于去噪图像嵌入，类型为 [`UNet2DConditionModel`]。
        # movq：从潜变量生成图像的 MoVQ 解码器，类型为 [`VQModel`]。
        # prior_prior：用于从文本嵌入近似图像嵌入的规范 unCLIP 先验，类型为 [`PriorTransformer`]。
        # prior_image_encoder：被冻结的图像编码器，类型为 [`CLIPVisionModelWithProjection`]。
        # prior_text_encoder：被冻结的文本编码器，类型为 [`CLIPTextModelWithProjection`]。
        # prior_tokenizer：类的分词器，类型为 [`CLIPTokenizer`]。
        # prior_scheduler：与 `prior` 结合生成图像嵌入的调度器，类型为 [`UnCLIPScheduler`]。
    """

    # 设定加载连接管道的标志为真
    _load_connected_pipes = True
    # 定义 CPU 卸载的模型序列
    model_cpu_offload_seq = "text_encoder->unet->movq->prior_prior->prior_image_encoder->prior_text_encoder"
    # 排除 CPU 卸载的部分
    _exclude_from_cpu_offload = ["prior_prior"]

    # 构造函数定义，接收多个参数以初始化类
    def __init__(
        # 文本编码器参数，类型为 MultilingualCLIP
        self,
        text_encoder: MultilingualCLIP,
        # 分词器参数，类型为 XLMRobertaTokenizer
        tokenizer: XLMRobertaTokenizer,
        # U-Net 参数，类型为 UNet2DConditionModel
        unet: UNet2DConditionModel,
        # 调度器参数，类型为 DDIMScheduler 或 DDPMScheduler
        scheduler: Union[DDIMScheduler, DDPMScheduler],
        # MoVQ 解码器参数，类型为 VQModel
        movq: VQModel,
        # 先验参数，类型为 PriorTransformer
        prior_prior: PriorTransformer,
        # 图像编码器参数，类型为 CLIPVisionModelWithProjection
        prior_image_encoder: CLIPVisionModelWithProjection,
        # 文本编码器参数，类型为 CLIPTextModelWithProjection
        prior_text_encoder: CLIPTextModelWithProjection,
        # 先验分词器参数，类型为 CLIPTokenizer
        prior_tokenizer: CLIPTokenizer,
        # 先验调度器参数，类型为 UnCLIPScheduler
        prior_scheduler: UnCLIPScheduler,
        # 图像处理器参数，类型为 CLIPImageProcessor
        prior_image_processor: CLIPImageProcessor,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册多个模块，传递各自的参数
        self.register_modules(
            # 文本编码器
            text_encoder=text_encoder,
            # 分词器
            tokenizer=tokenizer,
            # U-Net 模型
            unet=unet,
            # 调度器
            scheduler=scheduler,
            # MOVQ 模块
            movq=movq,
            # 先验模型的先验
            prior_prior=prior_prior,
            # 图像编码器
            prior_image_encoder=prior_image_encoder,
            # 先验文本编码器
            prior_text_encoder=prior_text_encoder,
            # 先验分词器
            prior_tokenizer=prior_tokenizer,
            # 先验调度器
            prior_scheduler=prior_scheduler,
            # 先验图像处理器
            prior_image_processor=prior_image_processor,
        )
        # 创建先验管道对象，封装先验相关模块
        self.prior_pipe = KandinskyPriorPipeline(
            # 传入先验模型
            prior=prior_prior,
            # 传入图像编码器
            image_encoder=prior_image_encoder,
            # 传入文本编码器
            text_encoder=prior_text_encoder,
            # 传入分词器
            tokenizer=prior_tokenizer,
            # 传入调度器
            scheduler=prior_scheduler,
            # 传入图像处理器
            image_processor=prior_image_processor,
        )
        # 创建解码器管道对象，封装解码所需模块
        self.decoder_pipe = KandinskyPipeline(
            # 传入文本编码器
            text_encoder=text_encoder,
            # 传入分词器
            tokenizer=tokenizer,
            # 传入 U-Net 模型
            unet=unet,
            # 传入调度器
            scheduler=scheduler,
            # 传入 MOVQ 模块
            movq=movq,
        )

    # 启用 Xformers 内存高效注意力机制的方法，支持可选的注意力操作
    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        # 调用解码器管道中的启用方法
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    # 启用顺序 CPU 卸载的方法，接收 GPU ID 参数
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        卸载所有模型（`unet`、`text_encoder`、`vae` 和 `safety checker` 状态字典）到 CPU，使用 🤗
        Accelerate，显著减少内存使用。模型被移动到 `torch.device('meta')`，仅在其特定子模块的
        `forward` 方法被调用时才在 GPU 上加载。卸载是基于子模块进行的。
        内存节省大于使用 `enable_model_cpu_offload`，但性能较低。
        """
        # 在先验管道中启用顺序 CPU 卸载
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        # 在解码器管道中启用顺序 CPU 卸载
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)

    # 进度条方法，接收可迭代对象和总数作为参数
    def progress_bar(self, iterable=None, total=None):
        # 在先验管道中设置进度条
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        # 在解码器管道中设置进度条
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)
        # 启用解码器管道中的模型 CPU 卸载
        self.decoder_pipe.enable_model_cpu_offload()

    # 设置进度条配置的方法，接收关键字参数
    def set_progress_bar_config(self, **kwargs):
        # 在先验管道中设置进度条配置
        self.prior_pipe.set_progress_bar_config(**kwargs)
        # 在解码器管道中设置进度条配置
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    # 使用 PyTorch 的无梯度上下文管理器，避免计算梯度
    @torch.no_grad()
    # 替换示例文档字符串的方法
    @replace_example_docstring(TEXT2IMAGE_EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法
    def __call__(
            # 输入提示，可以是单个字符串或字符串列表
            self,
            prompt: Union[str, List[str]],
            # 可选的负面提示，也可以是单个字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 推理步骤的数量，默认为100
            num_inference_steps: int = 100,
            # 指导比例，控制生成的图像与提示的相关性，默认为4.0
            guidance_scale: float = 4.0,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: int = 1,
            # 输出图像的高度，默认为512像素
            height: int = 512,
            # 输出图像的宽度，默认为512像素
            width: int = 512,
            # 先验指导比例，默认为4.0
            prior_guidance_scale: float = 4.0,
            # 先验推理步骤的数量，默认为25
            prior_num_inference_steps: int = 25,
            # 可选的生成器，用于控制随机数生成
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在变量，通常用于生成模型的输入
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为“pil”，指生成PIL图像
            output_type: Optional[str] = "pil",
            # 可选的回调函数，接收当前步骤和输出张量
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调的执行步数，默认为1
            callback_steps: int = 1,
            # 是否返回字典格式的结果，默认为True
            return_dict: bool = True,
# 定义一个结合管道类，用于使用 Kandinsky 进行图像到图像的生成
class KandinskyImg2ImgCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for image-to-image generation using Kandinsky

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        text_encoder ([`MultilingualCLIP`]):
            Frozen text-encoder.  # 冻结的文本编码器
        tokenizer ([`XLMRobertaTokenizer`]):
            Tokenizer of class  # 分词器类
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            A scheduler to be used in combination with `unet` to generate image latents.  # 用于与 `unet` 结合生成图像潜变量的调度器
        unet ([`UNet2DConditionModel`]):
            Conditional U-Net architecture to denoise the image embedding.  # 条件 U-Net 架构，用于去噪图像嵌入
        movq ([`VQModel`]):
            MoVQ Decoder to generate the image from the latents.  # MoVQ 解码器，从潜变量生成图像
        prior_prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.  # 经典的 unCLIP 先验，用于从文本嵌入近似图像嵌入
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.  # 冻结的图像编码器
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.  # 冻结的文本编码器
        prior_tokenizer (`CLIPTokenizer`):
             Tokenizer of class  # 分词器类
             [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        prior_scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.  # 用于与 `prior` 结合生成图像嵌入的调度器
    """

    _load_connected_pipes = True  # 指定是否加载连接的管道
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->prior_prior->" "text_encoder->unet->movq"  # 指定模型在 CPU 卸载时的顺序
    _exclude_from_cpu_offload = ["prior_prior"]  # 指定在 CPU 卸载时排除的组件

    def __init__(  # 初始化方法
        self,
        text_encoder: MultilingualCLIP,  # 文本编码器实例
        tokenizer: XLMRobertaTokenizer,  # 分词器实例
        unet: UNet2DConditionModel,  # U-Net 模型实例
        scheduler: Union[DDIMScheduler, DDPMScheduler],  # 调度器实例
        movq: VQModel,  # MoVQ 解码器实例
        prior_prior: PriorTransformer,  # 先验变换器实例
        prior_image_encoder: CLIPVisionModelWithProjection,  # 图像编码器实例
        prior_text_encoder: CLIPTextModelWithProjection,  # 文本编码器实例
        prior_tokenizer: CLIPTokenizer,  # 先验分词器实例
        prior_scheduler: UnCLIPScheduler,  # 先验调度器实例
        prior_image_processor: CLIPImageProcessor,  # 图像处理器实例
    # 定义构造函数的结束部分，调用父类的构造函数
        ):
            super().__init__()
    
            # 注册多个模块及其相应的组件
            self.register_modules(
                text_encoder=text_encoder,  # 文本编码器
                tokenizer=tokenizer,        # 分词器
                unet=unet,                  # UNet 模型
                scheduler=scheduler,        # 调度器
                movq=movq,                  # MOVQ 组件
                prior_prior=prior_prior,    # 先验模型
                prior_image_encoder=prior_image_encoder,  # 图像编码器
                prior_text_encoder=prior_text_encoder,    # 文本编码器
                prior_tokenizer=prior_tokenizer,          # 先验分词器
                prior_scheduler=prior_scheduler,          # 先验调度器
                prior_image_processor=prior_image_processor,  # 图像处理器
            )
            # 创建先验管道，使用多个先验组件
            self.prior_pipe = KandinskyPriorPipeline(
                prior=prior_prior,                          # 先验模型
                image_encoder=prior_image_encoder,          # 图像编码器
                text_encoder=prior_text_encoder,            # 文本编码器
                tokenizer=prior_tokenizer,                  # 分词器
                scheduler=prior_scheduler,                  # 调度器
                image_processor=prior_image_processor,      # 图像处理器
            )
            # 创建图像到图像的管道，使用多个解码组件
            self.decoder_pipe = KandinskyImg2ImgPipeline(
                text_encoder=text_encoder,  # 文本编码器
                tokenizer=tokenizer,        # 分词器
                unet=unet,                  # UNet 模型
                scheduler=scheduler,        # 调度器
                movq=movq,                  # MOVQ 组件
            )
    
        # 启用高效的 xformers 内存注意力机制
        def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
            self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)  # 在解码管道中启用该机制
    
        # 启用顺序 CPU 卸载，减少 GPU 内存使用
        def enable_sequential_cpu_offload(self, gpu_id=0):
            r"""  # 文档字符串，描述该方法的功能
            Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
            text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
            `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
            Note that offloading happens on a submodule basis. Memory savings are higher than with
            `enable_model_cpu_offload`, but performance is lower.
            """
            # 启用先验管道的顺序 CPU 卸载
            self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
            # 启用解码管道的顺序 CPU 卸载
            self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
    
        # 显示进度条，监控处理过程
        def progress_bar(self, iterable=None, total=None):
            self.prior_pipe.progress_bar(iterable=iterable, total=total)  # 在先验管道中显示进度条
            self.decoder_pipe.progress_bar(iterable=iterable, total=total)  # 在解码管道中显示进度条
            self.decoder_pipe.enable_model_cpu_offload()  # 启用解码管道的模型 CPU 卸载
    
        # 设置进度条的配置
        def set_progress_bar_config(self, **kwargs):
            self.prior_pipe.set_progress_bar_config(**kwargs)  # 设置先验管道的进度条配置
            self.decoder_pipe.set_progress_bar_config(**kwargs)  # 设置解码管道的进度条配置
    
        @torch.no_grad()  # 禁用梯度计算，减少内存使用
        @replace_example_docstring(IMAGE2IMAGE_EXAMPLE_DOC_STRING)  # 替换示例文档字符串
    # 定义可调用方法，允许实例对象像函数一样被调用
        def __call__(
            self,
            # 输入提示，可以是字符串或字符串列表
            prompt: Union[str, List[str]],
            # 输入图像，可以是张量、PIL 图像或它们的列表
            image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
            # 可选的负提示，可以是字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 推理步骤的数量，默认为 100
            num_inference_steps: int = 100,
            # 指导比例，默认为 4.0，用于控制生成内容的自由度
            guidance_scale: float = 4.0,
            # 每个提示生成的图像数量，默认为 1
            num_images_per_prompt: int = 1,
            # 强度参数，默认为 0.3，控制输入图像的影响程度
            strength: float = 0.3,
            # 输出图像的高度，默认为 512 像素
            height: int = 512,
            # 输出图像的宽度，默认为 512 像素
            width: int = 512,
            # 先前指导比例，默认为 4.0，用于先前生成步骤的控制
            prior_guidance_scale: float = 4.0,
            # 先前推理步骤的数量，默认为 25
            prior_num_inference_steps: int = 25,
            # 随机数生成器，可以是单个生成器或生成器列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 可选的潜在张量，用于传递预定义的潜在空间
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pil"，指定返回的图像格式
            output_type: Optional[str] = "pil",
            # 可选的回调函数，在每个步骤调用，接受步骤信息和当前生成的张量
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 回调函数调用的步骤间隔，默认为 1
            callback_steps: int = 1,
            # 返回结果的类型，默认为 True，表示返回字典格式
            return_dict: bool = True,
# 定义一个名为 KandinskyInpaintCombinedPipeline 的类，继承自 DiffusionPipeline 类
class KandinskyInpaintCombinedPipeline(DiffusionPipeline):
    """
    Combined Pipeline for generation using Kandinsky

    该模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取库为所有管道实现的通用方法
    （例如下载或保存、在特定设备上运行等）。

    参数：
        text_encoder ([`MultilingualCLIP`]):
            冻结的文本编码器。
        tokenizer ([`XLMRobertaTokenizer`]):
            令牌化器类。
        scheduler (Union[`DDIMScheduler`,`DDPMScheduler`]):
            用于与 `unet` 结合生成图像潜变量的调度器。
        unet ([`UNet2DConditionModel`]):
            用于去噪图像嵌入的条件 U-Net 架构。
        movq ([`VQModel`]):
            用于从潜变量生成图像的 MoVQ 解码器。
        prior_prior ([`PriorTransformer`]):
            近似文本嵌入的图像嵌入的典型 unCLIP 先验。
        prior_image_encoder ([`CLIPVisionModelWithProjection`]):
            冻结的图像编码器。
        prior_text_encoder ([`CLIPTextModelWithProjection`]):
            冻结的文本编码器。
        prior_tokenizer (`CLIPTokenizer`):
             令牌化器类
             [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer)。
        prior_scheduler ([`UnCLIPScheduler`]):
            用于与 `prior` 结合生成图像嵌入的调度器。
    """

    # 指示加载连接的管道
    _load_connected_pipes = True
    # 定义模型 CPU 卸载的顺序
    model_cpu_offload_seq = "prior_text_encoder->prior_image_encoder->prior_prior->text_encoder->unet->movq"
    # 指定从 CPU 卸载时要排除的部分
    _exclude_from_cpu_offload = ["prior_prior"]

    # 初始化方法，用于设置类的属性
    def __init__(
        # 文本编码器，类型为 MultilingualCLIP
        self,
        text_encoder: MultilingualCLIP,
        # 令牌化器，类型为 XLMRobertaTokenizer
        tokenizer: XLMRobertaTokenizer,
        # 条件 U-Net，类型为 UNet2DConditionModel
        unet: UNet2DConditionModel,
        # 调度器，类型为 DDIMScheduler 或 DDPMScheduler
        scheduler: Union[DDIMScheduler, DDPMScheduler],
        # MoVQ 解码器，类型为 VQModel
        movq: VQModel,
        # 先验转换器，类型为 PriorTransformer
        prior_prior: PriorTransformer,
        # 冻结的图像编码器，类型为 CLIPVisionModelWithProjection
        prior_image_encoder: CLIPVisionModelWithProjection,
        # 冻结的文本编码器，类型为 CLIPTextModelWithProjection
        prior_text_encoder: CLIPTextModelWithProjection,
        # 先验令牌化器，类型为 CLIPTokenizer
        prior_tokenizer: CLIPTokenizer,
        # 先验调度器，类型为 UnCLIPScheduler
        prior_scheduler: UnCLIPScheduler,
        # 图像处理器，类型为 CLIPImageProcessor
        prior_image_processor: CLIPImageProcessor,
    ):
        # 调用父类的初始化方法
        super().__init__()

        # 注册多个模块及其对应的参数
        self.register_modules(
            # 文本编码器模块
            text_encoder=text_encoder,
            # 分词器模块
            tokenizer=tokenizer,
            # UNet模块
            unet=unet,
            # 调度器模块
            scheduler=scheduler,
            # 移动质量模块
            movq=movq,
            # 先验模块
            prior_prior=prior_prior,
            # 先验图像编码器
            prior_image_encoder=prior_image_encoder,
            # 先验文本编码器
            prior_text_encoder=prior_text_encoder,
            # 先验分词器
            prior_tokenizer=prior_tokenizer,
            # 先验调度器
            prior_scheduler=prior_scheduler,
            # 先验图像处理器
            prior_image_processor=prior_image_processor,
        )
        # 初始化先验管道，封装多个模块
        self.prior_pipe = KandinskyPriorPipeline(
            # 传入先验模块
            prior=prior_prior,
            # 传入图像编码器
            image_encoder=prior_image_encoder,
            # 传入文本编码器
            text_encoder=prior_text_encoder,
            # 传入分词器
            tokenizer=prior_tokenizer,
            # 传入调度器
            scheduler=prior_scheduler,
            # 传入图像处理器
            image_processor=prior_image_processor,
        )
        # 初始化解码管道，封装多个模块
        self.decoder_pipe = KandinskyInpaintPipeline(
            # 传入文本编码器
            text_encoder=text_encoder,
            # 传入分词器
            tokenizer=tokenizer,
            # 传入 UNet 模块
            unet=unet,
            # 传入调度器
            scheduler=scheduler,
            # 传入移动质量模块
            movq=movq,
        )

    # 启用 xformers 的内存高效注意力机制
    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[Callable] = None):
        # 调用解码管道启用内存高效注意力
        self.decoder_pipe.enable_xformers_memory_efficient_attention(attention_op)

    # 启用顺序 CPU 离线处理
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        将所有模型转移到 CPU，显著减少内存使用。调用时，unet、
        text_encoder、vae 和安全检查器的状态字典保存到 CPU，然后转移到
        `torch.device('meta')，仅在其特定子模块的 `forward` 方法被调用时加载到 GPU。
        注意，离线处理是基于子模块的。相比于
        `enable_model_cpu_offload`，内存节省更高，但性能较低。
        """
        # 启用先验管道的顺序 CPU 离线处理
        self.prior_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        # 启用解码管道的顺序 CPU 离线处理
        self.decoder_pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)

    # 显示进度条
    def progress_bar(self, iterable=None, total=None):
        # 在先验管道中显示进度条
        self.prior_pipe.progress_bar(iterable=iterable, total=total)
        # 在解码管道中显示进度条
        self.decoder_pipe.progress_bar(iterable=iterable, total=total)
        # 启用解码管道的模型 CPU 离线处理
        self.decoder_pipe.enable_model_cpu_offload()

    # 设置进度条配置
    def set_progress_bar_config(self, **kwargs):
        # 在先验管道中设置进度条配置
        self.prior_pipe.set_progress_bar_config(**kwargs)
        # 在解码管道中设置进度条配置
        self.decoder_pipe.set_progress_bar_config(**kwargs)

    # 禁用梯度计算以节省内存
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(INPAINT_EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法，接受多个参数以生成图像
    def __call__(
        self,
        # 输入提示，可以是单个字符串或字符串列表
        prompt: Union[str, List[str]],
        # 输入图像，可以是张量、PIL 图像或它们的列表
        image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
        # 遮罩图像，用于指定图像的哪些部分将被处理，可以是张量、PIL 图像或它们的列表
        mask_image: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]],
        # 可选的负向提示，指定不希望生成的内容，可以是单个字符串或字符串列表
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 推理的步数，控制生成图像的细致程度，默认为 100
        num_inference_steps: int = 100,
        # 指导尺度，影响生成图像与提示之间的一致性，默认为 4.0
        guidance_scale: float = 4.0,
        # 每个提示生成的图像数量，默认为 1
        num_images_per_prompt: int = 1,
        # 生成图像的高度，默认为 512 像素
        height: int = 512,
        # 生成图像的宽度，默认为 512 像素
        width: int = 512,
        # 先前引导尺度，用于控制先前信息的影响，默认为 4.0
        prior_guidance_scale: float = 4.0,
        # 先前推理的步数，默认为 25
        prior_num_inference_steps: int = 25,
        # 可选的生成器，用于控制随机性，可以是单个生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 可选的潜在张量，用于指定初始潜在空间，默认为 None
        latents: Optional[torch.Tensor] = None,
        # 输出类型，指定生成图像的格式，默认为 "pil"
        output_type: Optional[str] = "pil",
        # 可选的回调函数，在生成过程中调用，接收步数和生成的张量
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        # 回调函数调用的步数间隔，默认为 1
        callback_steps: int = 1,
        # 是否返回字典格式的结果，默认为 True
        return_dict: bool = True,
```