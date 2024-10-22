# `.\diffusers\pipelines\kandinsky\pipeline_kandinsky_prior.py`

```py
# 版权信息，表明该代码的版权归 HuggingFace 团队所有
# 
# 根据 Apache 许可证，版本 2.0（“许可证”），
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下网址获得许可证的副本：
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# 除非适用法律要求或书面同意，软件按“现状”分发，
# 不提供任何形式的担保或条件，无论是明示或暗示。
# 查看许可证以了解特定语言适用的权限和
# 限制。

# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass
# 从 typing 模块导入 List, Optional 和 Union 类型
from typing import List, Optional, Union

# 导入 numpy 库，并简化为 np
import numpy as np
# 导入 PIL 库中的 Image 模块
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 transformers 库中导入 CLIP 相关的处理器和模型
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

# 从父级模块导入 PriorTransformer 类
from ...models import PriorTransformer
# 从父级模块导入 UnCLIPScheduler 类
from ...schedulers import UnCLIPScheduler
# 从父级模块导入多个工具函数
from ...utils import (
    BaseOutput,  # 基本输出类
    logging,     # 日志模块
    replace_example_docstring,  # 替换示例文档字符串的函数
)
# 从 torch_utils 模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从 pipeline_utils 模块导入 DiffusionPipeline 类
from ..pipeline_utils import DiffusionPipeline

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 示例文档字符串，包含用法示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyPipeline, KandinskyPriorPipeline
        >>> import torch

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior")
        >>> pipe_prior.to("cuda")

        >>> prompt = "red cat, 4k photo"
        >>> out = pipe_prior(prompt)
        >>> image_emb = out.image_embeds
        >>> negative_image_emb = out.negative_image_embeds

        >>> pipe = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1")
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     prompt,
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=negative_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=100,
        ... ).images

        >>> image[0].save("cat.png")
        ```py
"""

# 另一个示例文档字符串，未填充内容
EXAMPLE_INTERPOLATE_DOC_STRING = """


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
    # 示例用法
    Examples:
        ```py
        # 从 diffusers 库中导入 KandinskyPriorPipeline 和 KandinskyPipeline
        >>> from diffusers import KandinskyPriorPipeline, KandinskyPipeline
        # 从 diffusers.utils 导入 load_image 函数
        >>> from diffusers.utils import load_image
        # 导入 PIL 库
        >>> import PIL

        # 导入 torch 库
        >>> import torch
        # 从 torchvision 导入 transforms 模块
        >>> from torchvision import transforms

        # 加载预训练的 KandinskyPriorPipeline，设置数据类型为 float16
        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16
        ... )
        # 将管道移动到 GPU 设备
        >>> pipe_prior.to("cuda")

        # 从指定 URL 加载第一张图像
        >>> img1 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )

        # 从指定 URL 加载第二张图像
        >>> img2 = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/starry_night.jpeg"
        ... )

        # 定义图像和描述文本的列表
        >>> images_texts = ["a cat", img1, img2]
        # 定义每个图像的权重
        >>> weights = [0.3, 0.3, 0.4]
        # 使用管道进行插值，返回图像嵌入和零图像嵌入
        >>> image_emb, zero_image_emb = pipe_prior.interpolate(images_texts, weights)

        # 加载预训练的 KandinskyPipeline，设置数据类型为 float16
        >>> pipe = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16)
        # 将管道移动到 GPU 设备
        >>> pipe.to("cuda")

        # 使用管道生成图像，设置相关参数
        >>> image = pipe(
        ...     "",
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=zero_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=150,
        ... ).images[0]

        # 将生成的图像保存为文件 "starry_cat.png"
        >>> image.save("starry_cat.png")
"""
# 文档字符串，描述该模块的用途
@dataclass
# 装饰器，定义数据类以便更方便地管理属性
class KandinskyPriorPipelineOutput(BaseOutput):
    """
    Output class for KandinskyPriorPipeline.
    # 文档字符串，描述 KandinskyPriorPipeline 的输出类

    Args:
        image_embeds (`torch.Tensor`)
            clip image embeddings for text prompt
        # 描述图像嵌入的参数，供文本提示使用
        negative_image_embeds (`List[PIL.Image.Image]` or `np.ndarray`)
            clip image embeddings for unconditional tokens
        # 描述用于无条件标记的图像嵌入参数
    """

    # 定义输出类的属性，包括图像嵌入和负图像嵌入
    image_embeds: Union[torch.Tensor, np.ndarray]
    # 图像嵌入属性，可以是 PyTorch 张量或 NumPy 数组
    negative_image_embeds: Union[torch.Tensor, np.ndarray]
    # 负图像嵌入属性，也可以是 PyTorch 张量或 NumPy 数组


class KandinskyPriorPipeline(DiffusionPipeline):
    """
    Pipeline for generating image prior for Kandinsky
    # 文档字符串，描述生成 Kandinsky 图像先验的管道

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    # 说明该模型继承自 DiffusionPipeline，参考超类文档以了解通用方法
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`PriorTransformer`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        # 先验变换器，用于从文本嵌入近似图像嵌入
        image_encoder ([`CLIPVisionModelWithProjection`]):
            Frozen image-encoder.
        # 冻结的图像编码器
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        # 冻结的文本编码器
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        # CLIP 词元化器
        scheduler ([`UnCLIPScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
        # 调度器，用于结合先验生成图像嵌入
    """

    # 定义不进行 CPU 卸载的模块
    _exclude_from_cpu_offload = ["prior"]
    # 定义模型 CPU 卸载序列
    model_cpu_offload_seq = "text_encoder->prior"

    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: CLIPVisionModelWithProjection,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: UnCLIPScheduler,
        image_processor: CLIPImageProcessor,
    ):
        # 初始化方法，接受多个参数用于构建管道
        super().__init__()
        # 调用超类的初始化方法

        # 注册模块，方便后续管理
        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )

    @torch.no_grad()
    # 修饰函数，表示在此函数中不需要计算梯度
    @replace_example_docstring(EXAMPLE_INTERPOLATE_DOC_STRING)
    # 替换示例文档字符串，提供相关的例子
    def interpolate(
        self,
        images_and_prompts: List[Union[str, PIL.Image.Image, torch.Tensor]],
        # 输入图像和提示的列表，可以是字符串、PIL 图像或 PyTorch 张量
        weights: List[float],
        # 对应每个提示的权重列表
        num_images_per_prompt: int = 1,
        # 每个提示生成的图像数量，默认为 1
        num_inference_steps: int = 25,
        # 推理步骤的数量，默认为 25
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 随机数生成器，用于控制生成的随机性
        latents: Optional[torch.Tensor] = None,
        # 潜在变量，可选，默认为 None
        negative_prior_prompt: Optional[str] = None,
        # 负先验提示，默认为 None
        negative_prompt: str = "",
        # 负提示，默认为空字符串
        guidance_scale: float = 4.0,
        # 引导缩放因子，影响生成图像的风格
        device=None,
        # 指定设备，默认为 None
    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    # 准备潜在变量，依据输入形状、数据类型、设备等参数
    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        # 如果潜在变量为空，则生成随机张量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果潜在变量的形状不匹配，则引发错误
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            # 将潜在变量转移到指定设备
            latents = latents.to(device)
    
        # 将潜在变量乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回调整后的潜在变量
        return latents
    
    # 获取零嵌入，用于生成图像的初始状态
    def get_zero_embed(self, batch_size=1, device=None):
        # 如果未指定设备，则使用默认设备
        device = device or self.device
        # 创建一个零张量，形状为图像大小
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=device, dtype=self.image_encoder.dtype
        )
        # 通过图像编码器获取零图像的嵌入表示
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        # 将零图像嵌入重复指定的批大小
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        # 返回零图像嵌入
        return zero_image_emb
    
    # 编码提示的私有方法，处理输入提示和设备
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        # 在后续逻辑中会使用此方法进行提示编码
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        # 定义调用方法，处理输入提示
        def __call__(
            self,
            prompt: Union[str, List[str]],
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: int = 1,
            num_inference_steps: int = 25,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            guidance_scale: float = 4.0,
            output_type: Optional[str] = "pt",
            return_dict: bool = True,
```