# `.\diffusers\pipelines\kandinsky2_2\pipeline_kandinsky2_2_prior_emb2emb.py`

```py
# 导入类型提示模块，用于类型注释
from typing import List, Optional, Union

# 导入 PIL 库中的 Image 模块，用于图像处理
import PIL.Image
# 导入 PyTorch 库
import torch
# 从 transformers 库导入 CLIP 相关的模型和处理器
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

# 从本地模块导入 PriorTransformer 模型
from ...models import PriorTransformer
# 从本地模块导入 UnCLIPScheduler 调度器
from ...schedulers import UnCLIPScheduler
# 从本地工具模块导入日志记录和文档字符串替换功能
from ...utils import (
    logging,
    replace_example_docstring,
)
# 从本地工具模块导入随机张量生成函数
from ...utils.torch_utils import randn_tensor
# 从本地模块导入 KandinskyPriorPipelineOutput
from ..kandinsky import KandinskyPriorPipelineOutput
# 从本地模块导入 DiffusionPipeline
from ..pipeline_utils import DiffusionPipeline

# 创建一个日志记录器，使用模块的名称作为标识
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# 定义示例文档字符串，展示用法示例
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from diffusers import KandinskyV22Pipeline, KandinskyV22PriorEmb2EmbPipeline
        >>> import torch

        >>> pipe_prior = KandinskyPriorPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
        ... )
        >>> pipe_prior.to("cuda")

        >>> prompt = "red cat, 4k photo"
        >>> img = load_image(
        ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
        ...     "/kandinsky/cat.png"
        ... )
        >>> image_emb, nagative_image_emb = pipe_prior(prompt, image=img, strength=0.2).to_tuple()

        >>> pipe = KandinskyPipeline.from_pretrained(
        ...     "kandinsky-community/kandinsky-2-2-decoder, torch_dtype=torch.float16"
        ... )
        >>> pipe.to("cuda")

        >>> image = pipe(
        ...     image_embeds=image_emb,
        ...     negative_image_embeds=negative_image_emb,
        ...     height=768,
        ...     width=768,
        ...     num_inference_steps=100,
        ... ).images

        >>> image[0].save("cat.png")
        ```py
"""

# 定义插值示例文档字符串，内容为空
EXAMPLE_INTERPOLATE_DOC_STRING = """

``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
```py  # 结束插值示例文档字符串
``` 
    # 示例代码，演示如何使用 Kandinsky 模型进行图像处理
        Examples:
            ```py
            # 从 diffusers 库导入所需的类和函数
            >>> from diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22Pipeline
            >>> from diffusers.utils import load_image
            >>> import PIL
    
            # 导入 PyTorch 库
            >>> import torch
            >>> from torchvision import transforms
    
            # 加载 Kandinsky 的先验模型
            >>> pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
            ...     "kandinsky-community/kandinsky-2-2-prior", torch_dtype=torch.float16
            ... )
            # 将模型转移到 GPU
            >>> pipe_prior.to("cuda")
    
            # 加载第一张图片
            >>> img1 = load_image(
            ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            ...     "/kandinsky/cat.png"
            ... )
    
            # 加载第二张图片
            >>> img2 = load_image(
            ...     "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main"
            ...     "/kandinsky/starry_night.jpeg"
            ... )
    
            # 创建包含文本描述和图像的列表
            >>> images_texts = ["a cat", img1, img2]
            # 设置每个图像的权重
            >>> weights = [0.3, 0.3, 0.4]
            # 调用插值方法，生成图像嵌入和零图像嵌入
            >>> image_emb, zero_image_emb = pipe_prior.interpolate(images_texts, weights)
    
            # 加载 Kandinsky 的解码器模型
            >>> pipe = KandinskyV22Pipeline.from_pretrained(
            ...     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16
            ... )
            # 将模型转移到 GPU
            >>> pipe.to("cuda")
    
            # 使用图像嵌入生成新图像
            >>> image = pipe(
            ...     image_embeds=image_emb,
            ...     negative_image_embeds=zero_image_emb,
            ...     height=768,
            ...     width=768,
            ...     num_inference_steps=150,
            ... ).images[0]
    
            # 将生成的图像保存为文件
            >>> image.save("starry_cat.png")
            ```
"""
# 文档字符串，描述这个类的功能
class KandinskyV22PriorEmb2EmbPipeline(DiffusionPipeline):
    """
    # 文档字符串，描述生成图像先验的管道
    Pipeline for generating image prior for Kandinsky

    # 说明此模型继承自 [`DiffusionPipeline`]，并提到上级文档中的通用方法
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    # 构造函数参数说明
    Args:
        prior ([`PriorTransformer`]):
            # 用于近似文本嵌入生成图像嵌入的 canonical unCLIP prior
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            # 冻结的图像编码器
            Frozen image-encoder.
        text_encoder ([`CLIPTextModelWithProjection`]):
            # 冻结的文本编码器
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            # CLIPTokenizer 的分词器
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`UnCLIPScheduler`]):
            # 与 `prior` 结合使用的调度器，用于生成图像嵌入
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    # 定义 CPU 释放的顺序
    model_cpu_offload_seq = "text_encoder->image_encoder->prior"
    # 指定不参与 CPU 释放的模块
    _exclude_from_cpu_offload = ["prior"]

    # 构造函数，初始化各个组件
    def __init__(
        self,
        prior: PriorTransformer,
        image_encoder: CLIPVisionModelWithProjection,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        scheduler: UnCLIPScheduler,
        image_processor: CLIPImageProcessor,
    ):
        # 调用父类构造函数
        super().__init__()

        # 注册模块
        self.register_modules(
            prior=prior,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )

    # 获取时间步长的方法
    def get_timesteps(self, num_inference_steps, strength, device):
        # 计算初始时间步，确保不超过总时间步数
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        # 计算开始时间步，确保不小于零
        t_start = max(num_inference_steps - init_timestep, 0)
        # 获取指定时间步范围
        timesteps = self.scheduler.timesteps[t_start:]

        # 返回时间步和剩余推理步骤
        return timesteps, num_inference_steps - t_start

    # 该方法不计算梯度，用于插值
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_INTERPOLATE_DOC_STRING)
    def interpolate(
        # 输入图像及提示，类型可以是字符串、PIL图像或张量
        images_and_prompts: List[Union[str, PIL.Image.Image, torch.Tensor]],
        # 权重列表
        weights: List[float],
        # 每个提示生成的图像数量，默认为1
        num_images_per_prompt: int = 1,
        # 推理步骤数量，默认为25
        num_inference_steps: int = 25,
        # 随机生成器，默认为None
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 潜在张量，默认为None
        latents: Optional[torch.Tensor] = None,
        # 负面先验提示，默认为None
        negative_prior_prompt: Optional[str] = None,
        # 负面提示，默认为空字符串
        negative_prompt: str = "",
        # 引导尺度，默认为4.0
        guidance_scale: float = 4.0,
        # 设备，默认为None
        device=None,
    def _encode_image(
        # 编码图像的方法，支持张量或PIL图像列表
        self,
        image: Union[torch.Tensor, List[PIL.Image.Image]],
        # 设备类型
        device,
        # 每个提示的图像数量
        num_images_per_prompt,
    ):
        # 检查输入的图像是否为 PyTorch 张量
        if not isinstance(image, torch.Tensor):
            # 如果不是，则通过图像处理器将其转换为张量，并将数据类型和设备设置为合适的值
            image = self.image_processor(image, return_tensors="pt").pixel_values.to(
                dtype=self.image_encoder.dtype, device=device
            )

        # 使用图像编码器对图像进行编码，提取图像嵌入
        image_emb = self.image_encoder(image)["image_embeds"]  # B, D
        # 按每个提示的图像数量重复图像嵌入
        image_emb = image_emb.repeat_interleave(num_images_per_prompt, dim=0)
        # 将图像嵌入转移到指定的设备
        image_emb.to(device=device)

        # 返回处理后的图像嵌入
        return image_emb

    def prepare_latents(self, emb, timestep, batch_size, num_images_per_prompt, dtype, device, generator=None):
        # 将嵌入转移到指定设备并设置数据类型
        emb = emb.to(device=device, dtype=dtype)

        # 计算新的批量大小
        batch_size = batch_size * num_images_per_prompt

        # 初始化潜在变量
        init_latents = emb

        # 检查批量大小是否大于初始潜在变量的形状，并且能否均匀复制
        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # 计算每个提示需要额外的图像数量
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            # 复制初始潜在变量以匹配新的批量大小
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        # 检查批量大小是否大于初始潜在变量的形状且不能均匀复制
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            # 抛出错误，提示无法复制图像
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            # 如果批量大小小于或等于初始潜在变量的形状，则直接使用初始潜在变量
            init_latents = torch.cat([init_latents], dim=0)

        # 获取初始潜在变量的形状
        shape = init_latents.shape
        # 生成噪声张量
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # 获取潜在变量
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        # 将潜在变量分配给变量以便返回
        latents = init_latents

        # 返回潜在变量
        return latents

    # 从 KandinskyPriorPipeline 复制的函数，获取零嵌入
    def get_zero_embed(self, batch_size=1, device=None):
        # 如果没有指定设备，则使用默认设备
        device = device or self.device
        # 创建一个零图像，形状为 (1, 3, 高, 宽)
        zero_img = torch.zeros(1, 3, self.image_encoder.config.image_size, self.image_encoder.config.image_size).to(
            device=device, dtype=self.image_encoder.dtype
        )
        # 对零图像进行编码以获取零嵌入
        zero_image_emb = self.image_encoder(zero_img)["image_embeds"]
        # 按批量大小重复零嵌入
        zero_image_emb = zero_image_emb.repeat(batch_size, 1)
        # 返回零嵌入
        return zero_image_emb

    # 从 KandinskyPriorPipeline 复制的函数，编码提示
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        # 定义输入的提示，可以是字符串或字符串列表
        prompt: Union[str, List[str]],
        # 输入的图像，可以是张量、图像列表或 PIL 图像
        image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]],
        # 设置强度，默认为 0.3
        strength: float = 0.3,
        # 可选的负面提示
        negative_prompt: Optional[Union[str, List[str]]] = None,
        # 每个提示的图像数量，默认为 1
        num_images_per_prompt: int = 1,
        # 推理步骤数量，默认为 25
        num_inference_steps: int = 25,
        # 随机数生成器，可以是生成器或生成器列表
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # 引导缩放比例，默认为 4.0
        guidance_scale: float = 4.0,
        # 输出类型，默认为 "pt"
        output_type: Optional[str] = "pt",  # pt only
        # 返回字典的标志，默认为 True
        return_dict: bool = True,
```