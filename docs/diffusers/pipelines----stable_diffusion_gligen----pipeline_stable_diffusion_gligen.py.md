# `.\diffusers\pipelines\stable_diffusion_gligen\pipeline_stable_diffusion_gligen.py`

```py
# 版权所有 2024 GLIGEN 作者和 HuggingFace 团队，保留所有权利。
#
# 根据 Apache 许可证第 2.0 版（“许可证”）进行许可；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下地址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面协议同意，软件
# 根据许可证分发是按“原样”基础， 
# 不提供任何形式的保证或条件，无论是明示还是暗示。
# 有关许可证具体权限和
# 限制的详细信息，请参阅许可证。

import inspect  # 导入 inspect 模块，用于获取信息和检查对象
import warnings  # 导入 warnings 模块，用于发出警告信息
from typing import Any, Callable, Dict, List, Optional, Union  # 导入类型提示相关的类

import PIL.Image  # 导入 PIL.Image 模块，用于图像处理
import torch  # 导入 PyTorch 库，用于深度学习
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer  # 从 transformers 导入相关模型和处理器

from ...image_processor import VaeImageProcessor  # 从相对路径导入 VaeImageProcessor
from ...loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin  # 导入加载器混合类
from ...models import AutoencoderKL, UNet2DConditionModel  # 导入相关模型
from ...models.attention import GatedSelfAttentionDense  # 导入自定义的注意力模块
from ...models.lora import adjust_lora_scale_text_encoder  # 导入调整 Lora 模型的函数
from ...schedulers import KarrasDiffusionSchedulers  # 导入调度器类
from ...utils import (  # 导入实用工具函数和常量
    USE_PEFT_BACKEND,  # 导入用于 PEFT 后端的常量
    deprecate,  # 导入用于标记过时功能的装饰器
    logging,  # 导入日志记录模块
    replace_example_docstring,  # 导入替换示例文档字符串的函数
    scale_lora_layers,  # 导入用于缩放 Lora 层的函数
    unscale_lora_layers,  # 导入用于取消缩放 Lora 层的函数
)
from ...utils.torch_utils import randn_tensor  # 从工具模块导入生成随机张量的函数
from ..pipeline_utils import DiffusionPipeline, StableDiffusionMixin  # 从管道工具模块导入相关类
from ..stable_diffusion import StableDiffusionPipelineOutput  # 从稳定扩散模块导入输出类
from ..stable_diffusion.safety_checker import StableDiffusionSafetyChecker  # 导入安全检查器

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器，使用 pylint 禁用无效名称警告

EXAMPLE_DOC_STRING = """  # 定义示例文档字符串的多行字符串
    Examples:
        ```py
        >>> import torch  # 导入 PyTorch 库
        >>> from diffusers import StableDiffusionGLIGENPipeline  # 从 diffusers 库导入 StableDiffusionGLIGENPipeline 类
        >>> from diffusers.utils import load_image  # 从 diffusers.utils 导入 load_image 函数

        >>> # 在由边界框定义的区域插入由文本描述的对象
        >>> pipe = StableDiffusionGLIGENPipeline.from_pretrained(  # 从预训练模型加载 StableDiffusionGLIGENPipeline
        ...     "masterful/gligen-1-4-inpainting-text-box", variant="fp16", torch_dtype=torch.float16  # 指定模型名称和数据类型
        ... )
        >>> pipe = pipe.to("cuda")  # 将模型移动到 GPU 上

        >>> input_image = load_image(  # 从指定 URL 加载输入图像
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/livingroom_modern.png"  # 图像 URL
        ... )
        >>> prompt = "a birthday cake"  # 定义生成图像的文本提示
        >>> boxes = [[0.2676, 0.6088, 0.4773, 0.7183]]  # 定义边界框的位置
        >>> phrases = ["a birthday cake"]  # 定义要插入的对象描述

        >>> images = pipe(  # 调用管道生成图像
        ...     prompt=prompt,  # 传入文本提示
        ...     gligen_phrases=phrases,  # 传入要插入的描述
        ...     gligen_inpaint_image=input_image,  # 传入需要修复的图像
        ...     gligen_boxes=boxes,  # 传入边界框
        ...     gligen_scheduled_sampling_beta=1,  # 设定计划采样的 beta 值
        ...     output_type="pil",  # 输出类型设为 PIL 图像
        ...     num_inference_steps=50,  # 设定推理步骤数量
        ... ).images  # 获取生成的图像列表

        >>> images[0].save("./gligen-1-4-inpainting-text-box.jpg")  # 将生成的第一张图像保存为 JPEG 文件

        >>> # 生成由提示描述的图像，并在由边界框定义的区域插入由文本描述的对象
        >>> pipe = StableDiffusionGLIGENPipeline.from_pretrained(  # 从预训练模型加载另一个 StableDiffusionGLIGENPipeline
        ...     "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16  # 指定新模型名称和数据类型
        ... )
        >>> pipe = pipe.to("cuda")  # 将模型移动到 GPU 上

        >>> prompt = "a waterfall and a modern high speed train running through the tunnel in a beautiful forest with fall foliage"  # 定义新的生成图像的文本提示
        >>> boxes = [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]  # 定义多个边界框的位置
        >>> phrases = ["a waterfall", "a modern high speed train running through the tunnel"]  # 定义要插入的多个对象描述

        >>> images = pipe(  # 调用管道生成图像
        ...     prompt=prompt,  # 传入新的文本提示
        ...     gligen_phrases=phrases,  # 传入新的要插入的描述
        ...     gligen_boxes=boxes,  # 传入新的边界框
        ...     gligen_scheduled_sampling_beta=1,  # 设定计划采样的 beta 值
        ...     output_type="pil",  # 输出类型设为 PIL 图像
        ...     num_inference_steps=50,  # 设定推理步骤数量
        ... ).images  # 获取生成的图像列表

        >>> images[0].save("./gligen-1-4-generation-text-box.jpg")  # 将生成的第一张图像保存为 JPEG 文件
        ```py 
# 定义一个名为 StableDiffusionGLIGENPipeline 的类，继承自 DiffusionPipeline 和 StableDiffusionMixin
class StableDiffusionGLIGENPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    用于使用 Stable Diffusion 和基于语言的图像生成 (GLIGEN) 的文本到图像生成管道。

    该模型从 [`DiffusionPipeline`] 继承。有关库为所有管道实现的通用方法的文档，请检查超类文档（例如下载或保存、在特定设备上运行等）。

    参数：
        vae ([`AutoencoderKL`]):
            用于将图像编码和解码为潜在表示的变分自编码器 (VAE) 模型。
        text_encoder ([`~transformers.CLIPTextModel`]):
            冻结的文本编码器 ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14))。
        tokenizer ([`~transformers.CLIPTokenizer`]):
            用于对文本进行标记的 `CLIPTokenizer`。
        unet ([`UNet2DConditionModel`]):
            用于去噪编码图像潜在值的 `UNet2DConditionModel`。
        scheduler ([`SchedulerMixin`]):
            与 `unet` 结合使用的调度器，用于去噪编码图像潜在值。可以是
            [`DDIMScheduler`], [`LMSDiscreteScheduler`] 或 [`PNDMScheduler`] 之一。
        safety_checker ([`StableDiffusionSafetyChecker`]):
            分类模块，估计生成的图像是否可能被认为是冒犯性或有害的。
            有关模型潜在危害的更多详细信息，请参考 [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5)。
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            用于从生成图像中提取特征的 `CLIPImageProcessor`；作为 `safety_checker` 的输入。
    """

    # 定义可选组件列表，包括安全检查器和特征提取器
    _optional_components = ["safety_checker", "feature_extractor"]
    # 定义模型在 CPU 上卸载的顺序
    model_cpu_offload_seq = "text_encoder->unet->vae"
    # 定义从 CPU 卸载中排除的组件，安全检查器不被卸载
    _exclude_from_cpu_offload = ["safety_checker"]

    # 初始化方法，接受多个参数以配置管道
    def __init__(
        # VAE 模型，负责图像的编码和解码
        vae: AutoencoderKL,
        # 文本编码器，用于处理输入文本
        text_encoder: CLIPTextModel,
        # 用于对文本进行标记的分词器
        tokenizer: CLIPTokenizer,
        # 用于去噪图像的 UNet 模型
        unet: UNet2DConditionModel,
        # 调度器，控制去噪过程
        scheduler: KarrasDiffusionSchedulers,
        # 安全检查器，评估生成图像的潜在危害
        safety_checker: StableDiffusionSafetyChecker,
        # 特征提取器，处理生成的图像
        feature_extractor: CLIPImageProcessor,
        # 是否需要安全检查器的标志，默认为真
        requires_safety_checker: bool = True,
    # 初始化父类
        ):
            super().__init__()
    
            # 检查是否禁用安全检查器并且需要安全检查器时，记录警告信息
            if safety_checker is None and requires_safety_checker:
                logger.warning(
                    f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                    " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                    " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                    " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                    " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                    " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
                )
    
            # 检查是否定义特征提取器以便使用安全检查器，如果没有则抛出错误
            if safety_checker is not None and feature_extractor is None:
                raise ValueError(
                    "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                    " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
                )
    
            # 注册模型模块，包括 VAE、文本编码器、分词器、UNet、调度器、安全检查器和特征提取器
            self.register_modules(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
            # 计算 VAE 的缩放因子
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            # 创建图像处理器，设置为转换 RGB
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
            # 将配置中的安全检查器要求注册到对象
            self.register_to_config(requires_safety_checker=requires_safety_checker)
    
        # 从 StableDiffusionPipeline 中复制的编码提示函数
        def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            lora_scale: Optional[float] = None,
            **kwargs,
    ):
        # 定义一个弃用消息，提示用户该方法将来会被移除，并建议使用新方法
        deprecation_message = "`_encode_prompt()` is deprecated and it will be removed in a future version. Use `encode_prompt()` instead. Also, be aware that the output format changed from a concatenated tensor to a tuple."
        # 调用 deprecate 函数，记录该方法的弃用信息
        deprecate("_encode_prompt()", "1.0.0", deprecation_message, standard_warn=False)

        # 调用 encode_prompt 方法，获取与提示相关的嵌入元组
        prompt_embeds_tuple = self.encode_prompt(
            prompt=prompt,  # 输入提示
            device=device,  # 设备类型（如 CPU 或 GPU）
            num_images_per_prompt=num_images_per_prompt,  # 每个提示生成的图像数量
            do_classifier_free_guidance=do_classifier_free_guidance,  # 是否使用无分类器的引导
            negative_prompt=negative_prompt,  # 负面提示
            prompt_embeds=prompt_embeds,  # 提示嵌入
            negative_prompt_embeds=negative_prompt_embeds,  # 负面提示嵌入
            lora_scale=lora_scale,  # Lora 缩放因子
            **kwargs,  # 其他可选参数
        )

        # 将嵌入元组中的两个张量连接起来，便于后续兼容
        prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])

        # 返回连接后的提示嵌入
        return prompt_embeds

    # 从 StableDiffusionPipeline 复制的 encode_prompt 方法
    def encode_prompt(
        self,
        prompt,  # 输入提示
        device,  # 设备类型（如 CPU 或 GPU）
        num_images_per_prompt,  # 每个提示生成的图像数量
        do_classifier_free_guidance,  # 是否使用无分类器的引导
        negative_prompt=None,  # 负面提示（可选）
        prompt_embeds: Optional[torch.Tensor] = None,  # 提示嵌入（可选）
        negative_prompt_embeds: Optional[torch.Tensor] = None,  # 负面提示嵌入（可选）
        lora_scale: Optional[float] = None,  # Lora 缩放因子（可选）
        clip_skip: Optional[int] = None,  # 跳过剪辑层（可选）
    # 从 StableDiffusionPipeline 复制的 run_safety_checker 方法
    def run_safety_checker(self, image, device, dtype):
        # 检查是否存在安全检查器
        if self.safety_checker is None:
            has_nsfw_concept = None  # 如果没有，则设置无敏感内容标志为 None
        else:
            # 如果输入是张量，则处理图像为 PIL 格式
            if torch.is_tensor(image):
                feature_extractor_input = self.image_processor.postprocess(image, output_type="pil")
            else:
                # 否则将输入转换为 PIL 格式
                feature_extractor_input = self.image_processor.numpy_to_pil(image)
            # 提取特征并将其转换为指定设备的张量
            safety_checker_input = self.feature_extractor(feature_extractor_input, return_tensors="pt").to(device)
            # 运行安全检查器，检查图像是否含有 NSFW 内容
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)  # 提供图像和特征输入
            )
        # 返回处理后的图像及其 NSFW 内容标志
        return image, has_nsfw_concept
    # 定义一个方法，用于准备调度器步骤的额外参数
        def prepare_extra_step_kwargs(self, generator, eta):
            # 准备调度器步骤的额外参数，因为不同的调度器具有不同的参数签名
            # eta (η) 仅在 DDIMScheduler 中使用，对于其他调度器将被忽略
            # eta 对应于 DDIM 论文中的 η: https://arxiv.org/abs/2010.02502
            # 应在 [0, 1] 范围内
    
            # 检查调度器的步骤方法是否接受 eta 参数
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 初始化一个空字典，用于存放额外的步骤参数
            extra_step_kwargs = {}
            # 如果调度器接受 eta 参数，则将其添加到额外参数字典中
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            # 检查调度器的步骤方法是否接受 generator 参数
            accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
            # 如果调度器接受 generator 参数，则将其添加到额外参数字典中
            if accepts_generator:
                extra_step_kwargs["generator"] = generator
            # 返回准备好的额外参数字典
            return extra_step_kwargs
    
        # 定义一个方法，用于检查输入参数的有效性
        def check_inputs(
            self,
            prompt,  # 文本提示，用于生成内容
            height,  # 生成内容的高度
            width,   # 生成内容的宽度
            callback_steps,  # 回调步骤的频率
            gligen_phrases,  # 用于生成的短语
            gligen_boxes,    # 用于生成的框
            negative_prompt=None,  # 可选的负面提示，用于生成的限制
            prompt_embeds=None,    # 可选的提示嵌入，提前计算的文本表示
            negative_prompt_embeds=None,  # 可选的负面提示嵌入，提前计算的负面文本表示
    ):
        # 检查高度和宽度是否都是 8 的倍数，如果不是，则抛出值错误
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        # 检查 callback_steps 是否为正整数，若条件不满足则抛出值错误
        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # 检查 prompt 和 prompt_embeds 是否同时被定义，如果是则抛出值错误
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        # 检查 prompt 和 prompt_embeds 是否都未定义，如果是则抛出值错误
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        # 检查 prompt 是否为字符串或列表类型，如果不是则抛出值错误
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        # 检查 negative_prompt 和 negative_prompt_embeds 是否同时被定义，如果是则抛出值错误
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        # 检查如果提供了 prompt_embeds 和 negative_prompt_embeds，则它们的形状是否相同，如果不同则抛出值错误
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        # 检查 gligen_phrases 和 gligen_boxes 的长度是否相同，如果不同则抛出值错误
        if len(gligen_phrases) != len(gligen_boxes):
            raise ValueError(
                "length of `gligen_phrases` and `gligen_boxes` has to be same, but"
                f" got: `gligen_phrases` {len(gligen_phrases)} != `gligen_boxes` {len(gligen_boxes)}"
            )

    # 从 diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents 复制的代码
    # 准备潜变量，创建指定形状的随机噪声或处理已给定的潜变量
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        # 定义潜变量的形状，包括批大小、通道数和经过 VAE 缩放因子后的高度和宽度
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        # 检查生成器的类型，如果是列表且长度与批大小不匹配则抛出错误
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # 如果没有提供潜变量，则通过随机生成创建新的潜变量
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            # 如果提供了潜变量，将其转移到指定设备上
            latents = latents.to(device)

        # 按调度器所需的标准差缩放初始噪声
        latents = latents * self.scheduler.init_noise_sigma
        # 返回处理后的潜变量
        return latents

    # 启用或禁用自注意力模块的融合
    def enable_fuser(self, enabled=True):
        # 遍历 UNet 模块
        for module in self.unet.modules():
            # 如果模块是 GatedSelfAttentionDense 类型，则设置其启用状态
            if type(module) is GatedSelfAttentionDense:
                module.enabled = enabled

    # 从给定的框列表生成修复掩码
    def draw_inpaint_mask_from_boxes(self, boxes, size):
        # 创建一个全为 1 的掩码，大小与输入图像一致
        inpaint_mask = torch.ones(size[0], size[1])
        # 遍历每个框，更新掩码中的相应区域为 0
        for box in boxes:
            x0, x1 = box[0] * size[0], box[2] * size[0]  # 计算框的左和右边界
            y0, y1 = box[1] * size[1], box[3] * size[1]  # 计算框的上和下边界
            inpaint_mask[int(y0) : int(y1), int(x0) : int(x1)] = 0  # 将框内区域设置为 0
        # 返回修复掩码
        return inpaint_mask

    # 裁剪图像到指定的新宽度和高度
    def crop(self, im, new_width, new_height):
        # 获取图像的当前宽度和高度
        width, height = im.size
        # 计算裁剪区域的左、上、右、下边界
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        # 返回裁剪后的图像
        return im.crop((left, top, right, bottom))

    # 根据目标尺寸对图像进行中心裁剪
    def target_size_center_crop(self, im, new_hw):
        # 获取图像的当前宽度和高度
        width, height = im.size
        # 如果宽度和高度不相等，进行中心裁剪
        if width != height:
            im = self.crop(im, min(height, width), min(height, width))
        # 将图像调整为新的宽高，并使用高质量的重采样方法
        return im.resize((new_hw, new_hw), PIL.Image.LANCZOS)

    # 装饰器，禁用梯度计算以节省内存
    @torch.no_grad()
    # 替换示例文档字符串
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    # 定义可调用对象的 __call__ 方法，允许实例像函数一样被调用
        def __call__(
            # 提示信息，可以是单个字符串或字符串列表
            self,
            prompt: Union[str, List[str]] = None,
            # 图像高度，可选参数
            height: Optional[int] = None,
            # 图像宽度，可选参数
            width: Optional[int] = None,
            # 推理步骤的数量，默认为50
            num_inference_steps: int = 50,
            # 引导强度，默认为7.5
            guidance_scale: float = 7.5,
            # Gligen 调度采样的 beta 值，默认为0.3
            gligen_scheduled_sampling_beta: float = 0.3,
            # Gligen 相关短语，可选字符串列表
            gligen_phrases: List[str] = None,
            # Gligen 边界框，列表中包含浮点数列表，可选
            gligen_boxes: List[List[float]] = None,
            # Gligen 使用的图像，PIL.Image.Image 对象，可选
            gligen_inpaint_image: Optional[PIL.Image.Image] = None,
            # 负提示信息，可选字符串或字符串列表
            negative_prompt: Optional[Union[str, List[str]]] = None,
            # 每个提示生成的图像数量，默认为1
            num_images_per_prompt: Optional[int] = 1,
            # 采样的 ETA 值，默认为0.0
            eta: float = 0.0,
            # 随机数生成器，可选，可以是单个或多个生成器
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            # 潜在张量，可选
            latents: Optional[torch.Tensor] = None,
            # 提示嵌入张量，可选
            prompt_embeds: Optional[torch.Tensor] = None,
            # 负提示嵌入张量，可选
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            # 输出类型，默认为“pil”
            output_type: Optional[str] = "pil",
            # 是否返回字典格式的结果，默认为 True
            return_dict: bool = True,
            # 可选回调函数，用于在每个步骤执行
            callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
            # 每隔多少步骤调用回调，默认为1
            callback_steps: int = 1,
            # 跨注意力的额外参数，可选
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            # 可选的跳过剪辑参数
            clip_skip: Optional[int] = None,
```