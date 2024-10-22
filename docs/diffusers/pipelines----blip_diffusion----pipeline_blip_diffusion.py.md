# `.\diffusers\pipelines\blip_diffusion\pipeline_blip_diffusion.py`

```py
# 版权所有 2024 Salesforce.com, inc.  # 指明版权归属
# 版权所有 2024 The HuggingFace Team. All rights reserved. # 指明另一个版权归属
# 根据 Apache License 2.0 许可协议进行授权； # 说明代码的许可协议
# 除非符合许可协议，否则不可使用此文件。 # 指出使用条件
# 可以在以下地址获取许可协议的副本： # 提供许可协议的获取方式
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"提供。 # 指出软件不提供任何保证
# 请参见许可协议了解特定的权限和限制。 # 指出许可协议的内容
from typing import List, Optional, Union  # 从 typing 模块导入类型提示工具

import PIL.Image  # 导入 PIL 库的图像处理功能
import torch  # 导入 PyTorch 库
from transformers import CLIPTokenizer  # 从 transformers 导入 CLIPTokenizer

from ...models import AutoencoderKL, UNet2DConditionModel  # 从相对路径导入模型
from ...schedulers import PNDMScheduler  # 从相对路径导入调度器
from ...utils import (  # 从相对路径导入工具函数
    logging,  # 导入日志记录工具
    replace_example_docstring,  # 导入替换示例文档字符串的工具
)
from ...utils.torch_utils import randn_tensor  # 从工具模块导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput  # 从相对路径导入管道工具
from .blip_image_processing import BlipImageProcessor  # 从当前目录导入图像处理工具
from .modeling_blip2 import Blip2QFormerModel  # 从当前目录导入 Blip2 模型
from .modeling_ctx_clip import ContextCLIPTextModel  # 从当前目录导入 ContextCLIP 文本模型

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器实例，pylint 禁用无效名称警告

EXAMPLE_DOC_STRING = """  # 示例文档字符串，展示如何使用 BlipDiffusionPipeline
    Examples:  # 示例部分的开始
        ```py  # 开始代码块
        >>> from diffusers.pipelines import BlipDiffusionPipeline  # 导入 BlipDiffusionPipeline
        >>> from diffusers.utils import load_image  # 导入加载图像的工具
        >>> import torch  # 导入 PyTorch 库

        >>> blip_diffusion_pipe = BlipDiffusionPipeline.from_pretrained(  # 创建 BlipDiffusionPipeline 的实例
        ...     "Salesforce/blipdiffusion", torch_dtype=torch.float16  # 从预训练模型加载，设置数据类型为 float16
        ... ).to("cuda")  # 将模型转移到 GPU

        >>> cond_subject = "dog"  # 定义条件主题为“狗”
        >>> tgt_subject = "dog"  # 定义目标主题为“狗”
        >>> text_prompt_input = "swimming underwater"  # 定义文本提示输入

        >>> cond_image = load_image(  # 加载条件图像
        ...     "https://huggingface.co/datasets/ayushtues/blipdiffusion_images/resolve/main/dog.jpg"  # 图像的 URL
        ... )  # 结束加载图像的函数调用
        >>> guidance_scale = 7.5  # 设置引导尺度
        >>> num_inference_steps = 25  # 设置推理步骤数
        >>> negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"  # 定义负面提示

        >>> output = blip_diffusion_pipe(  # 调用管道生成输出
        ...     text_prompt_input,  # 传入文本提示输入
        ...     cond_image,  # 传入条件图像
        ...     cond_subject,  # 传入条件主题
        ...     tgt_subject,  # 传入目标主题
        ...     guidance_scale=guidance_scale,  # 传入引导尺度
        ...     num_inference_steps=num_inference_steps,  # 传入推理步骤数
        ...     neg_prompt=negative_prompt,  # 传入负面提示
        ...     height=512,  # 设置输出图像高度
        ...     width=512,  # 设置输出图像宽度
        ... ).images  # 获取生成的图像
        >>> output[0].save("image.png")  # 保存生成的第一张图像为 "image.png"
        ```py  # 结束代码块
"""

class BlipDiffusionPipeline(DiffusionPipeline):  # 定义 BlipDiffusionPipeline 类，继承自 DiffusionPipeline
    """
    Pipeline for Zero-Shot Subject Driven Generation using Blip Diffusion.  # 说明该管道用于零-shot 主题驱动生成

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the  # 指出该模型继承自 DiffusionPipeline，并建议查看超类文档以获取通用方法
    # 库实现所有管道的功能（例如下载或保存，在特定设备上运行等）

    Args:
        tokenizer ([`CLIPTokenizer`]):
            文本编码器的分词器
        text_encoder ([`ContextCLIPTextModel`]):
            用于编码文本提示的文本编码器
        vae ([`AutoencoderKL`]):
            VAE 模型，用于将潜在变量映射到图像
        unet ([`UNet2DConditionModel`]):
            条件 U-Net 架构，用于去噪图像嵌入
        scheduler ([`PNDMScheduler`]):
             与 `unet` 一起使用以生成图像潜在变量的调度器
        qformer ([`Blip2QFormerModel`]):
            QFormer 模型，用于从文本和图像中获取多模态嵌入
        image_processor ([`BlipImageProcessor`]):
            图像处理器，用于图像的预处理和后处理
        ctx_begin_pos (int, `optional`, defaults to 2):
            文本编码器中上下文标记的位置
    """

    # 定义模型的 CPU 卸载顺序
    model_cpu_offload_seq = "qformer->text_encoder->unet->vae"

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: ContextCLIPTextModel,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: PNDMScheduler,
        qformer: Blip2QFormerModel,
        image_processor: BlipImageProcessor,
        ctx_begin_pos: int = 2,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        # 调用父类构造函数
        super().__init__()

        # 注册模块，包括分词器、文本编码器、VAE、U-Net、调度器、QFormer 和图像处理器
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            qformer=qformer,
            image_processor=image_processor,
        )
        # 将上下文开始位置、均值和标准差注册到配置中
        self.register_to_config(ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)

    # 获取查询嵌入的方法，输入图像和源主题
    def get_query_embeddings(self, input_image, src_subject):
        # 使用 QFormer 获取图像输入和文本输入的嵌入，返回字典
        return self.qformer(image_input=input_image, text_input=src_subject, return_dict=False)

    # 从原始 Blip Diffusion 代码复制，指定目标主题并通过重复增强提示
    def _build_prompt(self, prompts, tgt_subjects, prompt_strength=1.0, prompt_reps=20):
        # 初始化一个空列表，用于存放构建的提示
        rv = []
        # 遍历每个提示和目标主题
        for prompt, tgt_subject in zip(prompts, tgt_subjects):
            # 构建包含目标主题的提示
            prompt = f"a {tgt_subject} {prompt.strip()}"
            # 一个技巧来增强提示的效果
            rv.append(", ".join([prompt] * int(prompt_strength * prompt_reps)))

        # 返回构建的提示列表
        return rv

    # 从 diffusers.pipelines.consistency_models.pipeline_consistency_models.ConsistencyModelPipeline.prepare_latents 复制的代码
    # 准备潜在变量，包含批量大小、通道数、高度和宽度等参数
        def prepare_latents(self, batch_size, num_channels, height, width, dtype, device, generator, latents=None):
            # 定义潜在变量的形状
            shape = (batch_size, num_channels, height, width)
            # 检查生成器是否为列表且长度与批量大小不匹配，抛出值错误
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
    
            # 如果未提供潜在变量，则生成新的随机潜在变量
            if latents is None:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            else:
                # 将提供的潜在变量转移到指定设备和数据类型
                latents = latents.to(device=device, dtype=dtype)
    
            # 将初始噪声缩放到调度器所需的标准差
            latents = latents * self.scheduler.init_noise_sigma
            # 返回处理后的潜在变量
            return latents
    
        # 编码提示，生成文本嵌入
        def encode_prompt(self, query_embeds, prompt, device=None):
            # 如果未指定设备，则使用执行设备
            device = device or self._execution_device
    
            # 获取最大长度，考虑查询嵌入的上下文
            max_len = self.text_encoder.text_model.config.max_position_embeddings
            max_len -= self.qformer.config.num_query_tokens
    
            # 将提示进行分词处理，并调整为最大长度
            tokenized_prompt = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
    
            # 获取查询嵌入的批量大小
            batch_size = query_embeds.shape[0]
            # 为每个样本设置上下文起始位置
            ctx_begin_pos = [self.config.ctx_begin_pos] * batch_size
    
            # 使用文本编码器获取文本嵌入
            text_embeddings = self.text_encoder(
                input_ids=tokenized_prompt.input_ids,
                ctx_embeddings=query_embeds,
                ctx_begin_pos=ctx_begin_pos,
            )[0]
    
            # 返回生成的文本嵌入
            return text_embeddings
    
        # 禁用梯度计算，并替换示例文档字符串
        @torch.no_grad()
        @replace_example_docstring(EXAMPLE_DOC_STRING)
        def __call__(
            # 定义调用方法的输入参数，包括提示、参考图像等
            prompt: List[str],
            reference_image: PIL.Image.Image,
            source_subject_category: List[str],
            target_subject_category: List[str],
            latents: Optional[torch.Tensor] = None,
            guidance_scale: float = 7.5,
            height: int = 512,
            width: int = 512,
            num_inference_steps: int = 50,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            neg_prompt: Optional[str] = "",
            prompt_strength: float = 1.0,
            prompt_reps: int = 20,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
```