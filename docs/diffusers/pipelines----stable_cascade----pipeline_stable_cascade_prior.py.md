# `.\diffusers\pipelines\stable_cascade\pipeline_stable_cascade_prior.py`

```py
# 版权所有 2024 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证，版本 2.0（“许可证”）授权；
# 除非遵循该许可证，否则您不得使用此文件。
# 您可以在以下位置获得许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律或书面协议另有约定，否则根据许可证分发的软件是按“原样”基础分发的，
# 不附带任何形式的保证或条件，无论是明示还是暗示。
# 有关许可证所涵盖权限和限制的具体语言，请参见许可证。

from dataclasses import dataclass  # 从 dataclasses 模块导入 dataclass 装饰器
from math import ceil  # 从 math 模块导入 ceil 函数，用于向上取整
from typing import Callable, Dict, List, Optional, Union  # 导入类型提示相关的类型

import numpy as np  # 导入 numpy 库并简化为 np
import PIL  # 导入 PIL 库用于图像处理
import torch  # 导入 PyTorch 库
from transformers import CLIPImageProcessor, CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection  # 从 transformers 模块导入 CLIP 相关的处理器和模型

from ...models import StableCascadeUNet  # 从当前包的模型模块导入 StableCascadeUNet
from ...schedulers import DDPMWuerstchenScheduler  # 从当前包的调度器模块导入 DDPMWuerstchenScheduler
from ...utils import BaseOutput, logging, replace_example_docstring  # 从当前包的工具模块导入 BaseOutput、logging 和 replace_example_docstring
from ...utils.torch_utils import randn_tensor  # 从当前包的 PyTorch 工具模块导入 randn_tensor 函数
from ..pipeline_utils import DiffusionPipeline  # 从上级包的管道工具模块导入 DiffusionPipeline

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 定义 DEFAULT_STAGE_C_TIMESTEPS 为线性空间的时间步列表
DEFAULT_STAGE_C_TIMESTEPS = list(np.linspace(1.0, 2 / 3, 20)) + list(np.linspace(2 / 3, 0.0, 11))[1:]

# 示例文档字符串，展示如何使用 StableCascadePriorPipeline
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableCascadePriorPipeline  # 从 diffusers 模块导入 StableCascadePriorPipeline

        >>> prior_pipe = StableCascadePriorPipeline.from_pretrained(  # 创建预训练的 StableCascadePriorPipeline 实例
        ...     "stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16  # 指定模型路径和数据类型
        ... ).to("cuda")  # 将管道移动到 CUDA 设备

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"  # 定义输入提示
        >>> prior_output = pipe(prompt)  # 使用管道生成图像输出
        ```py
"""

@dataclass  # 使用 dataclass 装饰器定义数据类
class StableCascadePriorPipelineOutput(BaseOutput):  # 定义 StableCascadePriorPipelineOutput 类，继承自 BaseOutput
    """
    WuerstchenPriorPipeline 的输出类。

    Args:
        image_embeddings (`torch.Tensor` or `np.ndarray`)  # 图像嵌入，表示文本提示的图像特征
            Prior image embeddings for text prompt
        prompt_embeds (`torch.Tensor`):  # 文本提示的嵌入
            Text embeddings for the prompt.
        negative_prompt_embeds (`torch.Tensor`):  # 负文本提示的嵌入
            Text embeddings for the negative prompt.
    """

    image_embeddings: Union[torch.Tensor, np.ndarray]  # 定义图像嵌入属性，类型为 torch.Tensor 或 np.ndarray
    prompt_embeds: Union[torch.Tensor, np.ndarray]  # 定义提示嵌入属性，类型为 torch.Tensor 或 np.ndarray
    prompt_embeds_pooled: Union[torch.Tensor, np.ndarray]  # 定义池化提示嵌入属性，类型为 torch.Tensor 或 np.ndarray
    negative_prompt_embeds: Union[torch.Tensor, np.ndarray]  # 定义负提示嵌入属性，类型为 torch.Tensor 或 np.ndarray
    negative_prompt_embeds_pooled: Union[torch.Tensor, np.ndarray]  # 定义池化负提示嵌入属性，类型为 torch.Tensor 或 np.ndarray

class StableCascadePriorPipeline(DiffusionPipeline):  # 定义 StableCascadePriorPipeline 类，继承自 DiffusionPipeline
    """
    生成 Stable Cascade 的图像先验的管道。

    此模型继承自 [`DiffusionPipeline`]。请查看超类文档以获取库为所有管道实现的通用方法（如下载或保存，运行在特定设备上等）。
    # 函数的参数说明
        Args:
            prior ([`StableCascadeUNet`]):  # 稳定级联生成网络，用于近似从文本和/或图像嵌入得到的图像嵌入。
                The Stable Cascade prior to approximate the image embedding from the text and/or image embedding.
            text_encoder ([`CLIPTextModelWithProjection`]):  # 冻结的文本编码器，用于处理文本输入。
                Frozen text-encoder
                ([laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)).
            feature_extractor ([`~transformers.CLIPImageProcessor`]):  # 从生成的图像中提取特征的模型，作为输入传递给图像编码器。
                Model that extracts features from generated images to be used as inputs for the `image_encoder`.
            image_encoder ([`CLIPVisionModelWithProjection`]):  # 冻结的 CLIP 图像编码器，用于处理图像输入。
                Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
            tokenizer (`CLIPTokenizer`):  # 处理文本的分词器，能够将文本转为模型可理解的格式。
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            scheduler ([`DDPMWuerstchenScheduler`]):  # 调度器，与 prior 结合使用，以生成图像嵌入。
                A scheduler to be used in combination with `prior` to generate image embedding.
            resolution_multiple ('float', *optional*, defaults to 42.67):  # 生成多张图像时的默认分辨率。
                Default resolution for multiple images generated.
        """
    
        unet_name = "prior"  # 设置 prior 的名称为 "prior"
        text_encoder_name = "text_encoder"  # 设置文本编码器的名称
        model_cpu_offload_seq = "image_encoder->text_encoder->prior"  # 定义模型的 CPU 卸载顺序
        _optional_components = ["image_encoder", "feature_extractor"]  # 定义可选组件列表
        _callback_tensor_inputs = ["latents", "text_encoder_hidden_states", "negative_prompt_embeds"]  # 定义回调张量输入列表
    
        def __init__(  # 初始化方法
            self,
            tokenizer: CLIPTokenizer,  # 传入的分词器对象
            text_encoder: CLIPTextModelWithProjection,  # 传入的文本编码器对象
            prior: StableCascadeUNet,  # 传入的生成网络对象
            scheduler: DDPMWuerstchenScheduler,  # 传入的调度器对象
            resolution_multiple: float = 42.67,  # 设置默认分辨率的参数
            feature_extractor: Optional[CLIPImageProcessor] = None,  # 可选的特征提取器对象
            image_encoder: Optional[CLIPVisionModelWithProjection] = None,  # 可选的图像编码器对象
        ) -> None:  # 方法返回类型
            super().__init__()  # 调用父类的初始化方法
            self.register_modules(  # 注册各个模块
                tokenizer=tokenizer,  # 注册分词器
                text_encoder=text_encoder,  # 注册文本编码器
                image_encoder=image_encoder,  # 注册图像编码器
                feature_extractor=feature_extractor,  # 注册特征提取器
                prior=prior,  # 注册生成网络
                scheduler=scheduler,  # 注册调度器
            )
            self.register_to_config(resolution_multiple=resolution_multiple)  # 将分辨率参数注册到配置中
    
        def prepare_latents(  # 准备潜在变量的方法
            self,  # 指向实例本身
            batch_size,  # 批处理的大小
            height,  # 图像的高度
            width,  # 图像的宽度
            num_images_per_prompt,  # 每个提示生成的图像数量
            dtype,  # 数据类型
            device,  # 设备类型（CPU/GPU）
            generator,  # 随机数生成器
            latents,  # 潜在变量
            scheduler  # 调度器
    ):
        # 定义潜在形状，包括每个提示的图像数量和批处理大小等信息
        latent_shape = (
            num_images_per_prompt * batch_size,
            self.prior.config.in_channels,
            ceil(height / self.config.resolution_multiple),
            ceil(width / self.config.resolution_multiple),
        )

        # 如果潜在变量为空，则随机生成一个张量
        if latents is None:
            latents = randn_tensor(latent_shape, generator=generator, device=device, dtype=dtype)
        else:
            # 检查潜在变量的形状是否与预期匹配，不匹配则引发错误
            if latents.shape != latent_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latent_shape}")
            # 将潜在变量转移到指定设备
            latents = latents.to(device)

        # 将潜在变量乘以调度器的初始噪声标准差
        latents = latents * scheduler.init_noise_sigma
        # 返回处理后的潜在变量
        return latents

    def encode_prompt(
        # 定义编码提示的函数，接收多个参数
        self,
        device,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance,
        prompt=None,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_pooled: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,
    def encode_image(self, images, device, dtype, batch_size, num_images_per_prompt):
        # 定义编码图像的函数，初始化图像嵌入列表
        image_embeds = []
        for image in images:
            # 提取特征并将图像转为张量形式
            image = self.feature_extractor(image, return_tensors="pt").pixel_values
            # 将图像张量移动到指定设备和数据类型
            image = image.to(device=device, dtype=dtype)
            # 编码图像并将结果嵌入添加到列表中
            image_embed = self.image_encoder(image).image_embeds.unsqueeze(1)
            image_embeds.append(image_embed)
        # 将所有图像嵌入按维度1拼接在一起
        image_embeds = torch.cat(image_embeds, dim=1)

        # 重复图像嵌入以匹配批处理大小和每个提示的图像数量
        image_embeds = image_embeds.repeat(batch_size * num_images_per_prompt, 1, 1)
        # 创建与图像嵌入形状相同的零张量作为负面图像嵌入
        negative_image_embeds = torch.zeros_like(image_embeds)

        # 返回正面和负面图像嵌入
        return image_embeds, negative_image_embeds

    def check_inputs(
        # 定义输入检查函数，接收多个可能为空的参数
        self,
        prompt,
        images=None,
        image_embeds=None,
        negative_prompt=None,
        prompt_embeds=None,
        prompt_embeds_pooled=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_pooled=None,
        callback_on_step_end_tensor_inputs=None,
    @property
    # 定义属性以获取引导比例
    def guidance_scale(self):
        return self._guidance_scale

    @property
    # 定义属性以判断是否使用无分类器引导
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    # 定义属性以获取时间步数
    def num_timesteps(self):
        return self._num_timesteps

    def get_timestep_ratio_conditioning(self, t, alphas_cumprod):
        # 定义获取时间步比率的条件函数
        s = torch.tensor([0.003])
        clamp_range = [0, 1]
        # 计算最小方差
        min_var = torch.cos(s / (1 + s) * torch.pi * 0.5) ** 2
        var = alphas_cumprod[t]
        # 将方差限制在指定范围内
        var = var.clamp(*clamp_range)
        s, min_var = s.to(var.device), min_var.to(var.device)
        # 计算并返回比率
        ratio = (((var * min_var) ** 0.5).acos() / (torch.pi * 0.5)) * (1 + s) - s
        return ratio

    @torch.no_grad()
    # 装饰器用于不计算梯度
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义一个可调用的类方法，处理输入的各种参数
        def __call__(
            # 提示文本，可以是单个字符串或字符串列表
            self,
            prompt: Optional[Union[str, List[str]]] = None,
            # 输入图像，可以是单个 Tensor、PIL 图像或它们的列表
            images: Union[torch.Tensor, PIL.Image.Image, List[torch.Tensor], List[PIL.Image.Image]] = None,
            # 输出图像的高度，默认值为 1024
            height: int = 1024,
            # 输出图像的宽度，默认值为 1024
            width: int = 1024,
            # 推理步骤的数量，默认值为 20
            num_inference_steps: int = 20,
            # 时间步列表，决定生成图像的时间步数
            timesteps: List[float] = None,
            # 引导尺度，用于调整生成图像的质量，默认值为 4.0
            guidance_scale: float = 4.0,
            # 负提示文本，可以是单个字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 预计算的提示嵌入，可以为 Tensor
            prompt_embeds: Optional[torch.Tensor] = None,
            # 预计算的池化提示嵌入，可以为 Tensor
            prompt_embeds_pooled: Optional[torch.Tensor] = None,
            # 预计算的负提示嵌入，可以为 Tensor
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 预计算的池化负提示嵌入，可以为 Tensor
            negative_prompt_embeds_pooled: Optional[torch.Tensor] = None,
            # 输入图像的嵌入，可以为 Tensor
            image_embeds: Optional[torch.Tensor] = None,
            # 每个提示生成的图像数量，默认值为 1
            num_images_per_prompt: Optional[int] = 1,
            # 随机数生成器，可以是单个生成器或生成器的列表
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在变量，可以为 Tensor
            latents: Optional[torch.Tensor] = None,
            # 输出类型，默认为 "pt"
            output_type: Optional[str] = "pt",
            # 是否返回字典格式的结果，默认值为 True
            return_dict: bool = True,
            # 每一步结束时调用的回调函数，接受步数、总步数和状态字典
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            # 在步骤结束时的张量输入回调列表，默认包括 "latents"
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
```